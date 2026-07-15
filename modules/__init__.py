"""Replaceable neural network modules."""

from modules.attention import ATTENTION_REGISTRY, build_attention
from modules.block import BLOCK_REGISTRY, build_block
from modules.mlp import MLP_REGISTRY, build_mlp
from modules.multimodal import (
    MULTIMODAL_FUSION_REGISTRY,
    MULTIMODAL_PROJECTOR_REGISTRY,
    MULTIMODAL_RESAMPLER_REGISTRY,
    build_multimodal_fusion,
    build_multimodal_projector,
    build_multimodal_resampler,
)
from modules.moe import MOE_REGISTRY, build_moe
from modules.norm import NORM_REGISTRY, build_norm
from modules.position import POSITION_ENCODING_REGISTRY, build_position_encoding
from modules.vision import VISION_ENCODER_REGISTRY, build_vision_encoder

__all__ = [
    "ATTENTION_REGISTRY",
    "BLOCK_REGISTRY",
    "MLP_REGISTRY",
    "MULTIMODAL_FUSION_REGISTRY",
    "MULTIMODAL_PROJECTOR_REGISTRY",
    "MULTIMODAL_RESAMPLER_REGISTRY",
    "MOE_REGISTRY",
    "NORM_REGISTRY",
    "POSITION_ENCODING_REGISTRY",
    "VISION_ENCODER_REGISTRY",
    "build_attention",
    "build_block",
    "build_mlp",
    "build_multimodal_fusion",
    "build_multimodal_projector",
    "build_multimodal_resampler",
    "build_moe",
    "build_norm",
    "build_position_encoding",
    "build_vision_encoder",
]
