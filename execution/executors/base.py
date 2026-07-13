"""Executor lifecycle interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from execution.specs import JobRecord, JobSpec


class Executor(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    @abstractmethod
    def submit(self, spec: JobSpec) -> JobRecord:
        raise NotImplementedError

    @abstractmethod
    def status(self, record: JobRecord) -> JobRecord:
        raise NotImplementedError

    @abstractmethod
    def logs(self, record: JobRecord, tail: int = 200) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, record: JobRecord) -> JobRecord:
        raise NotImplementedError

    def fetch(self, record: JobRecord, destination: str | Path) -> Path | None:
        del record, destination
        return None


def read_tail(path: str | Path, lines: int) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    content = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])
