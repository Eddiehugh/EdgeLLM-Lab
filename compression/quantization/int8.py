"""Symmetric reference INT8 quantization."""

from __future__ import annotations

import torch

from compression.quantization.base import (
    QuantizedTensor,
    TensorQuantizer,
    validate_quantization_input,
)
from compression.quantization.registry import QUANTIZER_REGISTRY
from core import Maturity, ProjectLevel


def _normalized_axis(axis: int, ndim: int) -> int:
    normalized = axis if axis >= 0 else ndim + axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis={axis} is invalid for a {ndim}D tensor")
    return normalized


@QUANTIZER_REGISTRY.register(
    "int8",
    "symmetric_int8",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("tensor_quantization", "symmetric", "int8", "per_channel"),
)
class SymmetricInt8Quantizer(TensorQuantizer):
    """Symmetric per-tensor or per-channel INT8 quantizer."""

    bits = 8

    def __init__(self, granularity: str = "channel", axis: int = 0):
        if granularity not in {"tensor", "channel"}:
            raise ValueError("INT8 granularity must be 'tensor' or 'channel'")
        self.granularity = granularity
        self.axis = axis

    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        validate_quantization_input(tensor)
        source = tensor.detach().float()
        if self.granularity == "tensor":
            scale = source.abs().max().clamp_min(1e-12) / 127
            quantized = torch.round(source / scale).clamp(-127, 127).to(torch.int8)
            axis = None
        else:
            axis = _normalized_axis(self.axis, source.ndim)
            reduce_dims = tuple(index for index in range(source.ndim) if index != axis)
            if reduce_dims:
                scale = source.abs().amax(dim=reduce_dims).clamp_min(1e-12) / 127
            else:
                scale = source.abs().clamp_min(1e-12) / 127
            view_shape = [1] * source.ndim
            view_shape[axis] = source.shape[axis]
            quantized = torch.round(source / scale.view(view_shape)).clamp(-127, 127)
            quantized = quantized.to(torch.int8)
        return QuantizedTensor(
            values=quantized,
            scale=scale,
            shape=tuple(tensor.shape),
            bits=8,
            original_dtype=tensor.dtype,
            axis=axis,
        )

    def dequantize(self, tensor: QuantizedTensor) -> torch.Tensor:
        if tensor.bits != 8:
            raise ValueError("SymmetricInt8Quantizer requires an 8-bit tensor")
        if tensor.axis is None:
            result = tensor.values.float() * tensor.scale
        else:
            view_shape = [1] * len(tensor.shape)
            view_shape[tensor.axis] = tensor.shape[tensor.axis]
            result = tensor.values.float() * tensor.scale.view(view_shape)
        return result.reshape(tensor.shape).to(tensor.original_dtype)


def quantize_int8(tensor: torch.Tensor):
    """Backward-compatible per-tensor helper returning values and scale."""

    result = SymmetricInt8Quantizer(granularity="tensor").quantize(tensor)
    return result.values, result.scale


def dequantize_int8(values: torch.Tensor, scale: torch.Tensor):
    return values.float() * scale
