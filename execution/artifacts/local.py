"""Filesystem artifact store."""

from __future__ import annotations

import shutil
from pathlib import Path

from execution.artifacts.base import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path = "artifacts/jobs"):
        self.root = Path(root).expanduser()

    def uri_for(self, job_id: str) -> str:
        return str((self.root / job_id).resolve())

    def publish(self, source: str | Path, job_id: str) -> str:
        source_path = Path(source).resolve()
        destination = (self.root / job_id).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path != destination:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
        return str(destination)

    def fetch(self, uri: str, destination: str | Path) -> Path:
        source = Path(uri).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {source}")
        target = Path(destination).expanduser().resolve()
        if source == target:
            return target
        if target.exists():
            raise FileExistsError(f"Artifact destination already exists: {target}")
        shutil.copytree(source, target)
        return target
