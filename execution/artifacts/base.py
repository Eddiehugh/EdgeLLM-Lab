"""Artifact store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def uri_for(self, job_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def publish(self, source: str | Path, job_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, uri: str, destination: str | Path) -> Path:
        raise NotImplementedError
