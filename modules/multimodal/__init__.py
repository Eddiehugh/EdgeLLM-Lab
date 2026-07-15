"""Multimodal projector, resampler, fusion, and tensor contracts."""

from modules.multimodal.fusion import (
    MULTIMODAL_FUSION_REGISTRY,
    build_multimodal_fusion,
)
from modules.multimodal.projector import (
    MULTIMODAL_PROJECTOR_REGISTRY,
    build_multimodal_projector,
)
from modules.multimodal.resampler import (
    MULTIMODAL_RESAMPLER_REGISTRY,
    build_multimodal_resampler,
)
from modules.multimodal.types import FusionOutput, ModalityFeatures

__all__ = [
    "FusionOutput",
    "MULTIMODAL_FUSION_REGISTRY",
    "MULTIMODAL_PROJECTOR_REGISTRY",
    "MULTIMODAL_RESAMPLER_REGISTRY",
    "ModalityFeatures",
    "build_multimodal_fusion",
    "build_multimodal_projector",
    "build_multimodal_resampler",
]
