"""S3-compatible artifact storage through fsspec/s3fs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from execution.artifacts.base import ArtifactStore


class S3ArtifactStore(ArtifactStore):
    def __init__(self, uri: str, storage_options: dict[str, Any] | None = None):
        if not uri.startswith("s3://"):
            raise ValueError("S3 artifact URI must start with s3://")
        self.root_uri = uri.rstrip("/")
        self.storage_options = dict(storage_options or {})

    @staticmethod
    def _fsspec():
        try:
            import fsspec
        except ImportError as exc:
            raise RuntimeError(
                "S3 artifacts require the 'cloud' extra: pip install -e '.[cloud]'"
            ) from exc
        return fsspec

    def uri_for(self, job_id: str) -> str:
        return f"{self.root_uri}/{job_id}"

    def publish(self, source: str | Path, job_id: str) -> str:
        source_path = Path(source).resolve()
        uri = self.uri_for(job_id)
        fs, remote_path = self._fsspec().core.url_to_fs(uri, **self.storage_options)
        fs.put(str(source_path), remote_path, recursive=True)
        return uri

    def fetch(self, uri: str, destination: str | Path) -> Path:
        target = Path(destination).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        fs, remote_path = self._fsspec().core.url_to_fs(uri, **self.storage_options)
        fs.get(remote_path, str(target), recursive=True)
        return target
