"""Shared state passed through an experiment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.registry import Registry

if TYPE_CHECKING:
    from experiments.run_store import Run


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageRecord:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentContext:
    """Runtime objects, metrics, and provenance for one experiment run."""

    config: dict[str, Any]
    run: "Run"
    environment: dict[str, Any]
    objects: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    component_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    status: str = "running"
    error: str | None = None

    def provide(self, name: str, value: Any) -> Any:
        self.objects[name] = value
        return value

    def require(self, name: str) -> Any:
        try:
            return self.objects[name]
        except KeyError as exc:
            raise KeyError(
                f"Pipeline object '{name}' is unavailable. "
                "Check stage order and stage requirements."
            ) from exc

    def track_component(
        self,
        role: str,
        registry: Registry[Any],
        selected_name: str,
    ) -> None:
        self.component_specs[role] = registry.describe(selected_name)

    def record_artifact(self, name: str, path: str | Path) -> None:
        self.artifacts[name] = str(Path(path).resolve())

    def finish(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.finished_at = utc_now()

    def manifest(self) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "run_id": self.run.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "environment": self.environment,
            "components": self.component_specs,
            "stages": [stage.to_dict() for stage in self.stages],
            "artifacts": dict(self.artifacts),
        }
