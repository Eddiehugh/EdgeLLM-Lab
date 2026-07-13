"""Metadata store interface for execution lifecycle records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from execution.specs import JobRecord


class MetadataStore(ABC):
    @abstractmethod
    def save(self, record: JobRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, job_id: str) -> JobRecord:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def job_directory(self, job_id: str) -> Path:
        raise NotImplementedError
