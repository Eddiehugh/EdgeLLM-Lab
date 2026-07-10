"""Quantization algorithms."""

from compression.quantization.registry import (
    QUANTIZER_REGISTRY,
    build_quantizer,
    load_builtin_quantizers,
)

__all__ = ["QUANTIZER_REGISTRY", "build_quantizer", "load_builtin_quantizers"]
