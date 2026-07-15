"""Group-wise packed INT4 reference quantization."""

from __future__ import annotations

from math import prod

import torch

from compression.quantization.base import (
    QuantizedTensor,
    TensorQuantizer,
    validate_quantization_input,
)
from compression.quantization.packing import pack_int4, unpack_int4
from compression.quantization.registry import QUANTIZER_REGISTRY
from core import Maturity, ProjectLevel


@QUANTIZER_REGISTRY.register(
    "int4",
    "groupwise_int4",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("tensor_quantization", "symmetric", "int4", "groupwise", "packed"),
)
class GroupwiseInt4Quantizer(TensorQuantizer):
    """Packed symmetric INT4 quantization over groups on the last axis."""

    bits = 4

    def __init__(self, group_size: int = 32):
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.group_size = int(group_size)

    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        if tensor.ndim == 0:
            raise ValueError("INT4 group quantization requires at least one dimension")
        validate_quantization_input(tensor)
        source = tensor.detach().float()
        width = source.shape[-1]
        rows = prod(source.shape[:-1]) if source.ndim > 1 else 1
        groups = (width + self.group_size - 1) // self.group_size
        padded_width = groups * self.group_size
        matrix = source.reshape(rows, width)
        if padded_width != width:
            matrix = torch.nn.functional.pad(matrix, (0, padded_width - width))
        grouped = matrix.reshape(rows, groups, self.group_size)
        scale = grouped.abs().amax(dim=-1).clamp_min(1e-12) / 7
        quantized = torch.round(grouped / scale.unsqueeze(-1)).clamp(-7, 7)
        quantized = quantized.to(torch.int8)
        return QuantizedTensor(
            values=pack_int4(quantized),
            scale=scale,
            shape=tuple(tensor.shape),
            bits=4,
            original_dtype=tensor.dtype,
            axis=tensor.ndim - 1,
            group_size=self.group_size,
            packed=True,
            padded_numel=quantized.numel(),
        )

    def dequantize(self, tensor: QuantizedTensor) -> torch.Tensor:
        if tensor.bits != 4 or not tensor.packed or tensor.group_size is None:
            raise ValueError("GroupwiseInt4Quantizer requires packed group-wise INT4")
        padded_numel = tensor.padded_numel
        if padded_numel is None:
            raise ValueError("Packed INT4 tensor is missing padded_numel")
        width = tensor.shape[-1]
        rows = prod(tensor.shape[:-1]) if len(tensor.shape) > 1 else 1
        groups = tensor.scale.shape[-1]
        values = unpack_int4(tensor.values, padded_numel)
        grouped = values.reshape(rows, groups, tensor.group_size).float()
        matrix = (grouped * tensor.scale.reshape(rows, groups, 1)).reshape(rows, -1)
        return matrix[:, :width].reshape(tensor.shape).to(tensor.original_dtype)
