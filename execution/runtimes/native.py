"""Native Python process runtime."""

from __future__ import annotations

from pathlib import Path

from execution.runtimes.base import Runtime


class NativeRuntime(Runtime):
    def worker_command(
        self,
        project_root: str | Path,
        workspace: str | Path,
        job_spec_path: str | Path,
    ) -> list[str]:
        del project_root, workspace
        return [
            self.spec.python,
            "-m",
            "execution.worker",
            "--job-spec",
            str(job_spec_path),
        ]
