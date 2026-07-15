"""Multimodal token resampler registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


MULTIMODAL_RESAMPLER_REGISTRY = Registry[nn.Module]("multimodal_resampler")


def build_multimodal_resampler(
    resampler_type: str | dict = "identity",
    **kwargs,
) -> nn.Module:
    return build_from_config(
        MULTIMODAL_RESAMPLER_REGISTRY,
        resampler_type,
        default_type="identity",
        **kwargs,
    )
