"""Norm registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


NORM_REGISTRY = Registry[nn.Module]("norm")


def build_norm(
    norm_type: str | dict = "layernorm",
    *,
    hidden_size: int,
    **kwargs,
) -> nn.Module:
    """Build a normalization module by name."""

    return build_from_config(
        NORM_REGISTRY,
        norm_type,
        default_type="layernorm",
        hidden_size=hidden_size,
        **kwargs,
    )
