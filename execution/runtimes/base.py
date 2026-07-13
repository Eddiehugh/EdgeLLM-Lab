"""Runtime abstraction used by executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from execution.specs import RuntimeSpec


class Runtime(ABC):
    def __init__(self, spec: RuntimeSpec):
        self.spec = spec

    @abstractmethod
    def worker_command(
        self,
        project_root: str | Path,
        workspace: str | Path,
        job_spec_path: str | Path,
    ) -> list[str]:
        raise NotImplementedError
