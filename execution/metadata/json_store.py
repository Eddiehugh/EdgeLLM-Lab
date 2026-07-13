"""Atomic local JSON metadata store."""

from __future__ import annotations

import json
from pathlib import Path

from execution.metadata.base import MetadataStore
from execution.specs import JobRecord


class JsonMetadataStore(MetadataStore):
    def __init__(self, root: str | Path = ".edgellm/jobs"):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_directory(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, record: JobRecord) -> None:
        path = self.job_directory(record.job_id) / "record.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def load(self, job_id: str) -> JobRecord:
        path = self.root / job_id / "record.json"
        if not path.exists():
            raise KeyError(f"Unknown job id: {job_id}")
        return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[JobRecord]:
        records = []
        for path in self.root.glob("*/record.json"):
            records.append(
                JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            )
        return sorted(records, key=lambda record: record.created_at, reverse=True)
