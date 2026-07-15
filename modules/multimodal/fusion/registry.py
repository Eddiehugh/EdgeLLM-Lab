"""Multimodal fusion registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


MULTIMODAL_FUSION_REGISTRY = Registry[nn.Module]("multimodal_fusion")


def build_multimodal_fusion(
    fusion_type: str | dict = "prefix",
    **kwargs,
) -> nn.Module:
    return build_from_config(
        MULTIMODAL_FUSION_REGISTRY,
        fusion_type,
        default_type="prefix",
        **kwargs,
    )
