"""Docker runtime for isolated local and SSH workers."""

from __future__ import annotations

import shlex
from pathlib import Path

from execution.runtimes.base import Runtime


class DockerRuntime(Runtime):
    def worker_command(
        self,
        project_root: str | Path,
        workspace: str | Path,
        job_spec_path: str | Path,
    ) -> list[str]:
        if not self.spec.image:
            raise ValueError("runtime.image is required for the Docker runtime")
        project = Path(project_root)
        job_dir = Path(workspace)
        spec_name = Path(job_spec_path).name
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{project}:/workspace/source",
            "-v",
            f"{job_dir}:/workspace/job",
            "-w",
            "/workspace/source",
        ]
        for option in self.spec.options.get("docker_args", []):
            command.append(str(option))
        worker = [
            self.spec.python,
            "-m",
            "execution.worker",
            "--job-spec",
            f"/workspace/job/{spec_name}",
        ]
        if self.spec.options.get("install_project", True):
            extras = ",".join(self.spec.options.get("extras", []))
            install_target = "/workspace/source"
            if extras:
                install_target += f"[{extras}]"
            script = (
                f"{shlex.quote(self.spec.python)} -m pip install -e "
                f"{shlex.quote(install_target)} && {shlex.join(worker)}"
            )
            command.extend([self.spec.image, "bash", "-lc", script])
        else:
            command.extend([self.spec.image, *worker])
        return command
