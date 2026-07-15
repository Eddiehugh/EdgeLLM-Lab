"""Generic SSH executor; AutoDL is represented as an SSH profile."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from execution.executors.base import Executor
from execution.runtimes import build_runtime
from execution.specs import JobRecord, JobSpec, JobState


class SSHExecutor(Executor):
    def _password_prefix(self) -> list[str]:
        if not self.config.get("password"):
            return []
        executable = shutil.which("sshpass")
        if executable is None:
            raise RuntimeError(
                "Password authentication requires sshpass. Install it with "
                "`brew install sshpass` on macOS."
            )
        return [executable, "-e"]

    def _subprocess_env(self) -> dict[str, str] | None:
        password = self.config.get("password")
        if not password:
            return None
        environment = os.environ.copy()
        environment["SSHPASS"] = str(password)
        return environment

    def _target(self) -> str:
        host = self.config.get("host")
        if not host:
            raise ValueError("execution.executor.host is required for SSH/AutoDL")
        user = self.config.get("user")
        return f"{user}@{host}" if user else str(host)

    def _ssh_options(self) -> list[str]:
        options: list[str] = []
        if self.config.get("port"):
            options.extend(["-p", str(self.config["port"])])
        if self.config.get("identity_file") and not self.config.get("password"):
            identity = Path(str(self.config["identity_file"])).expanduser()
            options.extend(["-i", str(identity)])
        options.extend(str(value) for value in self.config.get("ssh_options", []))
        return options

    def _scp_options(self) -> list[str]:
        options: list[str] = []
        if self.config.get("port"):
            options.extend(["-P", str(self.config["port"])])
        if self.config.get("identity_file") and not self.config.get("password"):
            identity = Path(str(self.config["identity_file"])).expanduser()
            options.extend(["-i", str(identity)])
        options.extend(str(value) for value in self.config.get("ssh_options", []))
        return options

    def probe(self) -> dict[str, str]:
        """Verify the endpoint and return basic remote identity information."""

        identity_file = self.config.get("identity_file")
        if identity_file and not self.config.get("password"):
            identity = Path(str(identity_file)).expanduser()
            if not identity.is_file():
                raise FileNotFoundError(
                    f"SSH identity file does not exist: {identity}. Create a key and "
                    "install its public key on the remote instance first."
                )
        output = self._ssh(
            "printf 'edgellm-connection-ok\\n'; "
            "printf 'hostname='; hostname; "
            "printf 'working_directory='; pwd"
        )
        return {"target": self._target(), "output": output}

    def _ssh(self, command: str, *, check: bool = True) -> str:
        result = subprocess.run(
            [
                *self._password_prefix(),
                "ssh",
                *self._ssh_options(),
                self._target(),
                command,
            ],
            check=check,
            text=True,
            capture_output=True,
            env=self._subprocess_env(),
            timeout=float(self.config.get("command_timeout", 120)),
        )
        return result.stdout.strip()

    def _remote_workspace(self, job_id: str, executor_type: str) -> str:
        default_root = (
            "/root/autodl-tmp/edgellm-jobs"
            if executor_type == "autodl"
            else "/tmp/edgellm-jobs"
        )
        root = str(self.config.get("remote_root", default_root)).rstrip("/")
        return f"{root}/{job_id}"

    def _shell_init(self) -> list[str]:
        value = self.config.get("shell_init", [])
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(command, str) and command.strip() for command in value
        ):
            raise TypeError("execution.executor.shell_init must contain shell commands")
        return list(value)

    @staticmethod
    def _require_source(spec: JobSpec) -> tuple[str, str]:
        if not spec.source.repo_url or not spec.source.revision:
            raise ValueError(
                "SSH execution requires source.repo_url and source.revision"
            )
        return spec.source.repo_url, spec.source.revision

    @staticmethod
    def _build_remote_launch(
        setup: list[str], command: list[str], remote_log: str
    ) -> str:
        foreground = "; ".join(setup)
        background = (
            f"nohup {shlex.join(command)} > {shlex.quote(remote_log)} "
            "2>&1 < /dev/null"
        )
        return (
            f"{foreground}; {background} & "
            "printf '\\n__EDGELLM_PID__=%s\\n' $!"
        )

    @staticmethod
    def _parse_remote_pid(output: str) -> str:
        match = re.search(r"(?:^|\n)__EDGELLM_PID__=(\d+)\s*$", output)
        if match is None:
            raise RuntimeError("Remote launch did not return an EdgeLLM worker PID")
        return match.group(1)

    def submit(self, spec: JobSpec) -> JobRecord:
        repo_url, revision = self._require_source(spec)
        local_workspace = Path(spec.workspace).expanduser().resolve()
        local_workspace.mkdir(parents=True, exist_ok=True)
        remote_workspace = self._remote_workspace(spec.job_id, spec.executor_type)
        remote_source = f"{remote_workspace}/source"
        remote_spec_path = f"{remote_workspace}/job-spec.json"
        remote_log = f"{remote_workspace}/worker.log"

        spec_data = spec.to_dict()
        spec_data["workspace"] = remote_workspace
        spec_data["source"]["project_root"] = remote_source
        remote_spec = JobSpec.from_dict(spec_data)
        local_spec_path = local_workspace / "remote-job-spec.json"
        local_spec_path.write_text(
            json.dumps(remote_spec.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        self._ssh(f"mkdir -p {shlex.quote(remote_workspace)}")
        subprocess.run(
            [
                *self._password_prefix(),
                "scp",
                *self._scp_options(),
                str(local_spec_path),
                f"{self._target()}:{remote_spec_path}",
            ],
            check=True,
            env=self._subprocess_env(),
            timeout=float(self.config.get("command_timeout", 120)),
        )

        runtime = build_runtime(remote_spec.runtime)
        worker_command = runtime.worker_command(
            remote_source, remote_workspace, remote_spec_path
        )
        environment = ["env"]
        environment.extend(f"{key}={value}" for key, value in remote_spec.env.items())
        command = environment + worker_command if remote_spec.env else worker_command

        setup = [
            "set -e",
            *self._shell_init(),
            f"git clone {shlex.quote(repo_url)} {shlex.quote(remote_source)}",
            f"git -C {shlex.quote(remote_source)} checkout --detach "
            f"{shlex.quote(revision)}",
        ]
        if remote_spec.runtime.type == "native":
            extras = ",".join(remote_spec.runtime.options.get("extras", []))
            install_target = remote_source + (f"[{extras}]" if extras else "")
            setup.append(
                f"{shlex.quote(remote_spec.runtime.python)} -m pip install -e "
                f"{shlex.quote(install_target)}"
            )
        setup.append(f"cd {shlex.quote(remote_source)}")
        launch_output = self._ssh(
            self._build_remote_launch(setup, command, remote_log)
        )
        pid = self._parse_remote_pid(launch_output)
        return JobRecord(
            job_id=spec.job_id,
            name=spec.name,
            executor_type=spec.executor_type,
            state=JobState.RUNNING,
            spec=remote_spec.to_dict(),
            provider_job_id=pid,
            log_path=remote_log,
            details={"pid": pid, "target": self._target()},
        )

    def _remote_result(self, record: JobRecord) -> dict[str, Any] | None:
        spec = JobSpec.from_dict(record.spec)
        result_path = f"{spec.workspace}/worker-result.json"
        output = self._ssh(
            f"test -f {shlex.quote(result_path)} && "
            f"cat {shlex.quote(result_path)} || true"
        )
        return json.loads(output) if output else None

    def status(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        result = self._remote_result(record)
        if result:
            return record.update(
                state=JobState(result["state"]),
                artifact_uri=result.get("artifact_uri", record.artifact_uri),
                message=result.get("error"),
                details={**record.details, "worker_result": result},
            )
        pid = shlex.quote(str(record.provider_job_id))
        alive = self._ssh(f"kill -0 {pid} 2>/dev/null && echo running || echo missing")
        if alive == "running":
            return record.update(state=JobState.RUNNING)
        return record.update(
            state=JobState.FAILED,
            message="Remote worker exited without writing worker-result.json",
        )

    def logs(self, record: JobRecord, tail: int = 200) -> str:
        path = shlex.quote(record.log_path or "")
        return self._ssh(f"test -f {path} && tail -n {int(tail)} {path} || true")

    def cancel(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        self._ssh(
            f"kill {shlex.quote(str(record.provider_job_id))} 2>/dev/null || true"
        )
        return record.update(state=JobState.CANCELLED, message="Cancelled over SSH")

    def fetch(self, record: JobRecord, destination: str | Path) -> Path | None:
        result = self._remote_result(record)
        if not result or not result.get("run_dir"):
            return None
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"Artifact destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *self._password_prefix(),
                "scp",
                *self._scp_options(),
                "-r",
                f"{self._target()}:{result['run_dir']}",
                str(target),
            ],
            check=True,
            env=self._subprocess_env(),
            timeout=float(self.config.get("fetch_timeout", 3600)),
        )
        return target
