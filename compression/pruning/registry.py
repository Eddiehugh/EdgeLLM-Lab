"""Pruner registry and factory."""

from __future__ import annotations

from typing import Any

from core.registry import Registry, build_from_config


PRUNER_REGISTRY = Registry("pruner")
_BUILTINS_LOADED = False


def load_builtin_pruners() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import compression.pruning.magnitude  # noqa: F401
    import compression.pruning.nm  # noqa: F401
    import compression.pruning.structured  # noqa: F401


def build_pruner(pruner_type: str | dict = "magnitude", **kwargs: Any):
    load_builtin_pruners()
    return build_from_config(
        PRUNER_REGISTRY,
        pruner_type,
        default_type="magnitude",
        **kwargs,
    )
