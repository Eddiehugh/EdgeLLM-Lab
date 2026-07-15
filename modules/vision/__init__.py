"""Vision-specific model components."""

from modules.vision.encoder import VISION_ENCODER_REGISTRY, build_vision_encoder

__all__ = ["VISION_ENCODER_REGISTRY", "build_vision_encoder"]
