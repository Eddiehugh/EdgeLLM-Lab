"""Position encoding registry and factory."""

from __future__ import annotations

from core.registry import Registry, build_from_config


POSITION_ENCODING_REGISTRY = Registry("position_encoding")


def build_position_encoding(position_type: str | dict = "rope", **kwargs):
    """Build a position encoding module by name."""

    return build_from_config(
        POSITION_ENCODING_REGISTRY,
        position_type,
        default_type="rope",
        **kwargs,
    )
