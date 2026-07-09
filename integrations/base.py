"""Base classes for external project integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IntegrationInfo:
    """Metadata that explains how an external project is used."""

    name: str
    purpose: str
    project_url: str
    modes: tuple[str, ...]
    capabilities: tuple[str, ...]
    recommended_first_step: str
    local_path: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntegrationAdapter:
    """Adapter boundary for reusing an external open-source project."""

    info: IntegrationInfo

    def __init__(
        self,
        local_path: str | None = None,
        external_root: str | Path = "external_projects",
    ):
        self.external_root = Path(external_root)
        if local_path is not None:
            self.info = IntegrationInfo(
                **{
                    **self.info.to_dict(),
                    "local_path": local_path,
                }
            )

    @property
    def name(self) -> str:
        return self.info.name

    def is_available(self) -> bool:
        """Return whether the optional local checkout exists."""

        return self.local_project_path().exists()

    def local_project_path(self) -> Path:
        """Return the external checkout path without importing it."""

        if self.info.local_path is not None:
            return Path(self.info.local_path)
        return self.external_root / self.name / "repo"

    def validate(self) -> dict[str, Any]:
        """Return adapter status without requiring the external dependency."""

        return {
            **self.info.to_dict(),
            "external_workspace": str((self.external_root / self.name).resolve()),
            "expected_repo_path": str(self.local_project_path().resolve()),
            "available": self.is_available(),
            "source_policy": (
                "External project source stays under external_projects/<name>/repo; "
                "only thin adapters live in integrations/<name>/."
            ),
        }

    def config_templates(self) -> dict[str, dict[str, Any]]:
        """Return optional local config snippets exposed by this adapter."""

        return {}

    def component_imports(self) -> tuple[str, ...]:
        """Return modules to import when the adapter registers components."""

        return ()
