"""MLP registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


MLP_REGISTRY = Registry[nn.Module]("mlp")


def build_mlp(
    mlp_type: str | dict = "gelu",
    *,
    hidden_size: int,
    intermediate_size: int,
    **kwargs,
) -> nn.Module:
    """Build an MLP module by name."""

    return build_from_config(
        MLP_REGISTRY,
        mlp_type,
        default_type="gelu",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        **kwargs,
    )
