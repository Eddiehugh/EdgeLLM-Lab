"""Reference quantization container for key/value cache tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compression.quantization.base import QuantizedTensor, TensorQuantizer
from compression.quantization.int8 import SymmetricInt8Quantizer


@dataclass(frozen=True)
class QuantizedKVCache:
    key: QuantizedTensor
    value: QuantizedTensor

    @property
    def storage_bytes(self) -> int:
        return self.key.storage_bytes + self.value.storage_bytes


class KVCacheQuantizer:
    """Quantize K/V tensors independently without assuming a cache layout."""

    def __init__(self, quantizer: TensorQuantizer | None = None):
        self.quantizer = quantizer or SymmetricInt8Quantizer(granularity="channel", axis=-1)

    def quantize(self, key: torch.Tensor, value: torch.Tensor) -> QuantizedKVCache:
        if key.shape != value.shape:
            raise ValueError("Key and value cache tensors must have the same shape")
        return QuantizedKVCache(
            key=self.quantizer.quantize(key),
            value=self.quantizer.quantize(value),
        )

    def dequantize(self, cache: QuantizedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.quantizer.dequantize(cache.key),
            self.quantizer.dequantize(cache.value),
        )
