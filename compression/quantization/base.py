"""Core tensor quantization contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod

import torch


def validate_quantization_input(tensor: torch.Tensor) -> None:
    """Reject inputs that cannot produce meaningful reference quantization."""

    if not tensor.is_floating_point():
        raise TypeError("Quantization input must use a floating-point dtype")
    if tensor.numel() == 0:
        raise ValueError("Quantization input must not be empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Quantization input must contain only finite values")


def tensor_storage_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class QuantizedTensor:
    """Quantized values plus all metadata needed for deterministic dequantization."""

    values: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    original_dtype: torch.dtype
    zero_point: torch.Tensor | None = None
    axis: int | None = None
    group_size: int | None = None
    packed: bool = False
    padded_numel: int | None = None

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def storage_bytes(self) -> int:
        return (
            tensor_storage_bytes(self.values)
            + tensor_storage_bytes(self.scale)
            + tensor_storage_bytes(self.zero_point)
        )


class TensorQuantizer(ABC):
    """Reference contract for readable tensor quantization algorithms."""

    bits: int

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        raise NotImplementedError

    @abstractmethod
    def dequantize(self, tensor: QuantizedTensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.quantize(tensor)


def quantization_error(
    reference: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float]:
    """Return common reconstruction error metrics."""

    if reference.shape != reconstructed.shape:
        raise ValueError("Quantization error tensors must have identical shapes")
    if reference.numel() == 0:
        raise ValueError("Quantization error tensors must not be empty")
    difference = (reference.float() - reconstructed.float()).abs()
    mse = torch.mean(difference.square())
    signal = torch.mean(reference.float().square()).clamp_min(1e-24)
    return {
        "mae": float(difference.mean()),
        "max_abs_error": float(difference.max()),
        "mse": float(mse),
        "relative_mse": float(mse / signal),
    }
