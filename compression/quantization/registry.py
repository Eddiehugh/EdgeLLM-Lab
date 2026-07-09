"""Quantizer registry and factory."""

from __future__ import annotations

from typing import Any

from core.registry import Registry, build_from_config


QUANTIZER_REGISTRY = Registry("quantizer")
_BUILTINS_LOADED = False


def _load_builtin_quantizers() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import compression.quantization.int8  # noqa: F401


def build_quantizer(quantizer_type: str | dict = "int8", **kwargs: Any):
    """Build a quantizer by name."""
    _load_builtin_quantizers()
    return build_from_config(
        QUANTIZER_REGISTRY,
        quantizer_type,
        default_type="int8",
        **kwargs,
    )
