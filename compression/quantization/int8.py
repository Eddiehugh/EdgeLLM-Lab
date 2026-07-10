"""Symmetric INT8 quantization."""

from __future__ import annotations

import torch

from compression.quantization.registry import QUANTIZER_REGISTRY
from core import Maturity, ProjectLevel


def quantize_int8(x: torch.Tensor):
    scale = x.abs().max().clamp_min(1e-12) / 127
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale


@QUANTIZER_REGISTRY.register(
    "int8",
    level=ProjectLevel.LEARN,
    maturity=Maturity.EXPERIMENTAL,
    capabilities=("tensor_quantization", "symmetric", "int8"),
)
class SymmetricInt8Quantizer:
    """Callable symmetric INT8 tensor quantizer."""

    def quantize(self, x: torch.Tensor):
        return quantize_int8(x)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        return dequantize_int8(q, scale)

    def __call__(self, x: torch.Tensor):
        return self.quantize(x)
