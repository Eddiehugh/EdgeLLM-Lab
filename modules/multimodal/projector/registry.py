"""Multimodal projector registry and factory."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config


MULTIMODAL_PROJECTOR_REGISTRY = Registry[nn.Module]("multimodal_projector")


def build_multimodal_projector(
    projector_type: str | dict = "linear",
    *,
    input_size: int,
    output_size: int,
    **kwargs,
) -> nn.Module:
    return build_from_config(
        MULTIMODAL_PROJECTOR_REGISTRY,
        projector_type,
        default_type="linear",
        input_size=input_size,
        output_size=output_size,
        **kwargs,
    )
