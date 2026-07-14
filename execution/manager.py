"""Execution control plane coordinating providers, metadata, and artifacts."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from execution.artifacts import build_artifact_store
from execution.executors import build_executor
from execution.metadata import JsonMetadataStore, MetadataStore
from execution.profiles import ConnectionProfileStore
from execution.specs import (
    ArtifactSpec,
    CommandSpec,
    JobRecord,
    JobSpec,
    JobState,
    RuntimeSpec,
    SourceSpec,
    WorkloadSpec,
)
from experiments import normalize_experiment_config


REMOTE_EXECUTORS = {
    "ssh",
    "autodl",
    "huggingface_jobs",
    "hf_jobs",
    "clearml",
    "colab",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "experiment"


class RunManager:
    def __init__(
        self,
        metadata_store: MetadataStore | None = None,
        project_root: str | Path | None = None,
        connection_store: ConnectionProfileStore | None = None,
    ):
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self.metadata = metadata_store or JsonMetadataStore(
            self.project_root / ".edgellm" / "jobs"
        )
        self.connections = connection_store or ConnectionProfileStore(
            self.project_root / ".edgellm" / "connections.json"
        )

    def _git(self, *args: str) -> str | None:
        return self._git_at(self.project_root, *args)

    @staticmethod
    def _git_at(project_root: Path, *args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _https_repo_url(repo_url: Any, prefer_https: bool) -> Any:
        if (
            prefer_https
            and isinstance(repo_url, str)
            and repo_url.startswith("git@github.com:")
        ):
            return "https://github.com/" + repo_url[len("git@github.com:") :]
        return repo_url

    def _source_spec(
        self, source_config: Mapping[str, Any], executor_type: str
    ) -> SourceSpec:
        source = dict(source_config)
        repo_url = source.get("repo_url") or self._git("remote", "get-url", "origin")
        revision = source.get("revision") or self._git("rev-parse", "HEAD")
        repo_url = self._https_repo_url(
            repo_url,
            executor_type in REMOTE_EXECUTORS and source.get("prefer_https", True),
        )
        require_clean = bool(source.get("require_clean", True))
        if executor_type in REMOTE_EXECUTORS:
            if not repo_url or not revision:
                raise ValueError(
                    "Remote execution requires a Git repo_url and revision under "
                    "execution.source"
                )
            if require_clean and self._git("status", "--porcelain"):
                raise ValueError(
                    "Remote execution requires a clean Git worktree so the selected "
                    "revision exactly matches local code. Commit changes first or set "
                    "execution.source.require_clean=false deliberately."
                )
        return SourceSpec(
            repo_url=str(repo_url) if repo_url else None,
            revision=str(revision) if revision else None,
            project_root=str(self.project_root),
        )

    @staticmethod
    def _command_spec(value: Any, field_name: str) -> CommandSpec:
        if isinstance(value, (list, tuple)):
            argv = value
            skip_if_exists = None
        elif isinstance(value, Mapping):
            argv = value.get("argv")
            skip_if_exists = value.get("skip_if_exists")
        else:
            raise TypeError(f"{field_name} must be an argv list or mapping")
        if not isinstance(argv, (list, tuple)) or not argv:
            raise ValueError(f"{field_name}.argv must be a non-empty list")
        return CommandSpec(
            argv=tuple(str(argument) for argument in argv),
            skip_if_exists=(
                str(skip_if_exists) if skip_if_exists is not None else None
            ),
        )

    def _workload_spec(
        self, workload_config: Mapping[str, Any], executor_type: str
    ) -> WorkloadSpec:
        workload = dict(workload_config)
        workload_type = str(workload.pop("type", "experiment"))
        if workload_type == "experiment":
            return WorkloadSpec()
        if workload_type != "external_project":
            raise ValueError(
                "execution.workload.type must be 'experiment' or 'external_project'"
            )

        integration = str(workload.pop("integration", "external"))
        source_value = workload.pop("source", {})
        if not isinstance(source_value, Mapping):
            raise TypeError("execution.workload.source must be a mapping")
        source_config = dict(source_value)
        local_path_value = source_config.pop(
            "local_path", f"external_projects/{integration}"
        )
        local_path = Path(str(local_path_value)).expanduser()
        if not local_path.is_absolute():
            local_path = self.project_root / local_path
        local_path = local_path.resolve()
        repo_url = source_config.pop("repo_url", None) or self._git_at(
            local_path, "remote", "get-url", "origin"
        )
        revision = source_config.pop("revision", None) or self._git_at(
            local_path, "rev-parse", "HEAD"
        )
        require_clean = bool(source_config.pop("require_clean", True))
        prefer_https = bool(source_config.pop("prefer_https", True))
        if source_config:
            unknown = ", ".join(sorted(source_config))
            raise ValueError(f"Unknown execution.workload.source fields: {unknown}")
        repo_url = self._https_repo_url(
            repo_url, executor_type in REMOTE_EXECUTORS and prefer_https
        )
        if not repo_url or not revision:
            raise ValueError(
                "External workloads require a Git repo_url and revision; configure "
                "execution.workload.source or provide a local Git checkout"
            )
        if require_clean and (local_path / ".git").exists():
            if self._git_at(local_path, "status", "--porcelain"):
                raise ValueError(
                    f"External workload checkout is dirty: {local_path}. Commit and "
                    "push the change before cloud submission, or deliberately set "
                    "execution.workload.source.require_clean=false."
                )

        setup_value = workload.pop("setup", [])
        if not isinstance(setup_value, list):
            raise TypeError("execution.workload.setup must be a list")
        setup = tuple(
            self._command_spec(command, f"execution.workload.setup[{index}]")
            for index, command in enumerate(setup_value)
        )
        command_value = workload.pop("command", None)
        if command_value is None:
            raise ValueError("execution.workload.command is required")
        command = self._command_spec(command_value, "execution.workload.command")
        artifacts_value = workload.pop("artifacts", [])
        if not isinstance(artifacts_value, list):
            raise TypeError("execution.workload.artifacts must be a list")
        working_directory = str(workload.pop("working_directory", "."))
        if workload:
            unknown = ", ".join(sorted(workload))
            raise ValueError(f"Unknown execution.workload fields: {unknown}")
        return WorkloadSpec(
            type=workload_type,
            integration=integration,
            source=SourceSpec(
                repo_url=str(repo_url),
                revision=str(revision),
                project_root=str(local_path),
            ),
            setup=setup,
            command=command,
            working_directory=working_directory,
            artifacts=tuple(str(path) for path in artifacts_value),
        )

    @staticmethod
    def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
        value = config.get(key, {})
        if not isinstance(value, Mapping):
            raise TypeError(f"execution.{key} must be a mapping")
        return dict(value)

    def build_spec(self, config: Mapping[str, Any]) -> JobSpec:
        raw_execution = config.get("execution", {})
        if not isinstance(raw_execution, Mapping):
            raise TypeError("execution must be a mapping")
        raw_workload = raw_execution.get("workload", {})
        if not isinstance(raw_workload, Mapping):
            raise TypeError("execution.workload must be a mapping")
        workload_type = str(raw_workload.get("type", "experiment"))
        experiment_config = (
            normalize_experiment_config(config)
            if workload_type == "experiment"
            else copy.deepcopy(dict(config))
        )
        execution = experiment_config.get("execution", {})
        if not isinstance(execution, Mapping):
            raise TypeError("execution must be a mapping")

        executor_config = self._section(execution, "executor")
        executor_type = str(executor_config.pop("type", "local"))
        executor_config = self.connections.resolve(executor_config)
        runtime_config = self._section(execution, "runtime")
        runtime_type = str(runtime_config.pop("type", "native"))
        default_python = sys.executable if executor_type == "local" else "python3"
        runtime = RuntimeSpec(
            type=runtime_type,
            image=runtime_config.pop("image", None),
            python=str(runtime_config.pop("python", default_python)),
            options=runtime_config,
        )
        artifact_config = self._section(execution, "artifact_store")
        artifact_type = str(artifact_config.pop("type", "local"))
        artifact_store = ArtifactSpec(type=artifact_type, config=artifact_config)
        artifact_extra = {
            "huggingface_hub": "hf",
            "hf_hub": "hf",
            "s3": "s3",
        }.get(artifact_type)
        if artifact_extra:
            runtime_options = dict(runtime.options)
            extras = [str(value) for value in runtime_options.get("extras", [])]
            if artifact_extra not in extras:
                extras.append(artifact_extra)
            runtime_options["extras"] = extras
            runtime = RuntimeSpec(
                type=runtime.type,
                image=runtime.image,
                python=runtime.python,
                options=runtime_options,
            )
        if executor_type in {"huggingface_jobs", "hf_jobs", "clearml"}:
            if artifact_type == "local":
                raise ValueError(
                    f"Executor '{executor_type}' requires a durable remote artifact "
                    "store (huggingface_hub or s3), not local"
                )

        experiment = dict(experiment_config.get("experiment", {}))
        name = str(
            experiment.get("name", experiment_config.get("name", "experiment"))
        )
        job_id = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{_slugify(name)}-{uuid.uuid4().hex[:6]}"
        )
        workspace = self.metadata.job_directory(job_id) / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        env_config = self._section(execution, "env")
        env = {str(key): str(value) for key, value in env_config.items()}
        source = self._source_spec(self._section(execution, "source"), executor_type)
        workload = self._workload_spec(raw_workload, executor_type)
        return JobSpec(
            job_id=job_id,
            name=name,
            experiment_config=copy.deepcopy(experiment_config),
            executor_type=executor_type,
            executor_config=executor_config,
            runtime=runtime,
            artifact_store=artifact_store,
            source=source,
            workspace=str(workspace),
            workload=workload,
            env=env,
        )

    def submit(self, config: Mapping[str, Any]) -> JobRecord:
        spec = self.build_spec(config)
        executor = build_executor(spec.executor_type, spec.executor_config)
        record = executor.submit(spec)
        record.artifact_uri = build_artifact_store(spec.artifact_store).uri_for(
            spec.job_id
        )
        self.metadata.save(record)
        return record

    def status(self, job_id: str) -> JobRecord:
        record = self.metadata.load(job_id)
        spec = JobSpec.from_dict(record.spec)
        executor = build_executor(record.executor_type, spec.executor_config)
        record = executor.status(record)
        self.metadata.save(record)
        return record

    def logs(self, job_id: str, tail: int = 200) -> str:
        record = self.metadata.load(job_id)
        spec = JobSpec.from_dict(record.spec)
        return build_executor(record.executor_type, spec.executor_config).logs(
            record, tail=tail
        )

    def cancel(self, job_id: str) -> JobRecord:
        record = self.metadata.load(job_id)
        spec = JobSpec.from_dict(record.spec)
        record = build_executor(
            record.executor_type, spec.executor_config
        ).cancel(record)
        self.metadata.save(record)
        return record

    def fetch(self, job_id: str, destination: str | Path) -> Path:
        record = self.status(job_id)
        if record.state != JobState.COMPLETED:
            raise RuntimeError(
                f"Job '{job_id}' is {record.state.value}; artifacts are fetched only "
                "after completion"
            )
        spec = JobSpec.from_dict(record.spec)
        executor = build_executor(record.executor_type, spec.executor_config)
        provider_result = executor.fetch(record, destination)
        if provider_result is not None:
            return provider_result
        if not record.artifact_uri:
            raise RuntimeError(f"Job '{job_id}' has no artifact URI")
        return build_artifact_store(spec.artifact_store).fetch(
            record.artifact_uri, destination
        )

    def list(self) -> list[JobRecord]:
        return self.metadata.list()

    def wait(
        self, job_id: str, poll_interval: float = 1.0, timeout: float | None = None
    ) -> JobRecord:
        started = time.monotonic()
        while True:
            record = self.status(job_id)
            if record.state.terminal or record.state == JobState.PREPARED:
                return record
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for job '{job_id}'")
            time.sleep(poll_interval)
