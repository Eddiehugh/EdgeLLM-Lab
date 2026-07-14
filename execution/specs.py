"""Serializable contracts shared by execution backends and workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    PREPARED = "prepared"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


@dataclass(frozen=True)
class SourceSpec:
    repo_url: str | None = None
    revision: str | None = None
    project_root: str = "."

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceSpec":
        return cls(**dict(data))


@dataclass(frozen=True)
class RuntimeSpec:
    type: str = "native"
    image: str | None = None
    python: str = "python3"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeSpec":
        return cls(**dict(data))


@dataclass(frozen=True)
class ArtifactSpec:
    type: str = "local"
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactSpec":
        return cls(**dict(data))


@dataclass(frozen=True)
class CommandSpec:
    """One argv-based command in an external workload."""

    argv: tuple[str, ...]
    skip_if_exists: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandSpec":
        values = dict(data)
        values["argv"] = tuple(str(value) for value in values["argv"])
        return cls(**values)


@dataclass(frozen=True)
class WorkloadSpec:
    """Provider-neutral description of the code a worker executes."""

    type: str = "experiment"
    integration: str | None = None
    source: SourceSpec | None = None
    setup: tuple[CommandSpec, ...] = ()
    command: CommandSpec | None = None
    working_directory: str = "."
    artifacts: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkloadSpec":
        values = dict(data)
        source = values.get("source")
        values["source"] = SourceSpec.from_dict(source) if source else None
        values["setup"] = tuple(
            CommandSpec.from_dict(command) for command in values.get("setup", ())
        )
        command = values.get("command")
        values["command"] = CommandSpec.from_dict(command) if command else None
        values["artifacts"] = tuple(str(path) for path in values.get("artifacts", ()))
        return cls(**values)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    name: str
    experiment_config: dict[str, Any]
    executor_type: str
    executor_config: dict[str, Any]
    runtime: RuntimeSpec
    artifact_store: ArtifactSpec
    source: SourceSpec
    workspace: str
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    env: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobSpec":
        values = dict(data)
        values["runtime"] = RuntimeSpec.from_dict(values["runtime"])
        values["artifact_store"] = ArtifactSpec.from_dict(values["artifact_store"])
        values["source"] = SourceSpec.from_dict(values["source"])
        values["workload"] = WorkloadSpec.from_dict(values.get("workload", {}))
        return cls(**values)


@dataclass
class JobRecord:
    job_id: str
    name: str
    executor_type: str
    state: JobState
    spec: dict[str, Any]
    provider_job_id: str | None = None
    artifact_uri: str | None = None
    log_path: str | None = None
    url: str | None = None
    message: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def update(self, **changes: Any) -> "JobRecord":
        for key, value in changes.items():
            setattr(self, key, value)
        self.updated_at = utc_now()
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRecord":
        values = dict(data)
        values["state"] = JobState(values["state"])
        return cls(**values)
