"""Attention registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


ATTENTION_REGISTRY = Registry[nn.Module]("attention")


def build_attention(
    attention_type: str | dict = "mha",
    *,
    hidden_size: int,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """Build an Attention module by name."""

    return build_from_config(
        ATTENTION_REGISTRY,
        attention_type,
        default_type="mha",
        hidden_size=hidden_size,
        num_heads=num_heads,
        **kwargs,
    )
