"""Block registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


BLOCK_REGISTRY = Registry[nn.Module]("block")


def build_block(
    block_type: str | dict = "transformer",
    *,
    hidden_size: int,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """Build a Transformer block by name."""

    return build_from_config(
        BLOCK_REGISTRY,
        block_type,
        default_type="transformer",
        hidden_size=hidden_size,
        num_heads=num_heads,
        **kwargs,
    )
