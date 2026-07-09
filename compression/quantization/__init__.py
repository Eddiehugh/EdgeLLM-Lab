"""Quantization algorithms."""

from compression.quantization.registry import QUANTIZER_REGISTRY, build_quantizer

__all__ = ["QUANTIZER_REGISTRY", "build_quantizer"]
