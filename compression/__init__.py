"""Model compression utilities."""

from compression.quantization import QUANTIZER_REGISTRY, build_quantizer

__all__ = ["QUANTIZER_REGISTRY", "build_quantizer"]
