"""Small helpers shared by built-in experiment stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def selected_name(config: Any, default: str) -> str:
    if config is None:
        return default
    if isinstance(config, str):
        return config
    if isinstance(config, Mapping):
        return str(config.get("type", config.get("name", default)))
    raise TypeError(f"Component selector must be a string or mapping: {config!r}")
