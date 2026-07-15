"""Vision encoder registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


VISION_ENCODER_REGISTRY = Registry[nn.Module]("vision_encoder")


def build_vision_encoder(
    encoder_type: str | dict = "patch_transformer",
    **kwargs,
) -> nn.Module:
    return build_from_config(
        VISION_ENCODER_REGISTRY,
        encoder_type,
        default_type="patch_transformer",
        **kwargs,
    )
