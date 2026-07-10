"""Structured metadata for replaceable framework components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ProjectLevel(str, Enum):
    """The project layer in which a component primarily belongs."""

    LEARN = "level_1"
    EXPERIMENT = "level_2"
    WORK = "level_3"


class Maturity(str, Enum):
    """Implementation maturity exposed to configs, reports, and users."""

    PLANNED = "planned"
    EXPERIMENTAL = "experimental"
    VERIFIED = "verified"
    PRODUCTION = "production"


@dataclass(frozen=True)
class ComponentSpec:
    """Machine-readable description of one registered implementation."""

    name: str
    component_type: str
    description: str = ""
    level: ProjectLevel = ProjectLevel.LEARN
    maturity: Maturity = Maturity.EXPERIMENTAL
    capabilities: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    source: str | None = None

    def supports(self, *capabilities: str) -> bool:
        available = set(self.capabilities)
        return all(capability in available for capability in capabilities)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["level"] = self.level.value
        data["maturity"] = self.maturity.value
        return data
