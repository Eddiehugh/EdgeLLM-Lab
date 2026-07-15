"""Portable weight-only quantized Linear reference module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.quantization.base import (
    QuantizedTensor,
    TensorQuantizer,
    tensor_storage_bytes,
)


class ReferenceQuantizedLinear(nn.Module):
    """Store quantized weights and dequantize them for a reference forward pass.

    This module validates formats and model conversion. It does not claim latency
    acceleration; optimized backends should consume the same exported state through
    dedicated adapters.
    """

    inference_only = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantized_weight: QuantizedTensor,
        quantizer: TensorQuantizer,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.quantizer = quantizer
        self.weight_shape = tuple(quantized_weight.shape)
        self.weight_bits = int(quantized_weight.bits)
        self.weight_dtype = quantized_weight.original_dtype
        self.weight_axis = quantized_weight.axis
        self.group_size = quantized_weight.group_size
        self.packed = quantized_weight.packed
        self.padded_numel = quantized_weight.padded_numel
        self.register_buffer("qweight", quantized_weight.values)
        self.register_buffer("weight_scale", quantized_weight.scale)
        self.register_buffer("weight_zero_point", quantized_weight.zero_point)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_float(
        cls, module: nn.Linear, quantizer: TensorQuantizer
    ) -> "ReferenceQuantizedLinear":
        return cls(
            module.in_features,
            module.out_features,
            quantizer.quantize(module.weight),
            quantizer,
            module.bias,
        )

    def quantized_weight(self) -> QuantizedTensor:
        return QuantizedTensor(
            values=self.qweight,
            scale=self.weight_scale,
            shape=self.weight_shape,
            bits=self.weight_bits,
            original_dtype=self.weight_dtype,
            zero_point=self.weight_zero_point,
            axis=self.weight_axis,
            group_size=self.group_size,
            packed=self.packed,
            padded_numel=self.padded_numel,
        )

    def dequantized_weight(self) -> torch.Tensor:
        return self.quantizer.dequantize(self.quantized_weight())

    @property
    def weight_storage_bytes(self) -> int:
        return (
            tensor_storage_bytes(self.qweight)
            + tensor_storage_bytes(self.weight_scale)
            + tensor_storage_bytes(self.weight_zero_point)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight().to(device=inputs.device, dtype=inputs.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.weight_bits}, packed={self.packed}"
        )
