"""MoE registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


MOE_REGISTRY = Registry[nn.Module]("moe")


def build_moe(moe_type: str | dict = "topk_router", **kwargs) -> nn.Module:
    """Build an MoE component by name."""

    return build_from_config(
        MOE_REGISTRY,
        moe_type,
        default_type="topk_router",
        **kwargs,
    )
