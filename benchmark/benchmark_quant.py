"""Compression correctness and sparsity benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from benchmark.registry import BENCHMARK_REGISTRY
from compression.quantization.base import quantization_error
from core import Maturity, ProjectLevel


@BENCHMARK_REGISTRY.register(
    "quantization_error",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("compression", "quantization", "numerical_error"),
)
class QuantizationErrorBenchmark:
    """Measure reconstruction error between reference and quantized tensors."""

    def __call__(
        self,
        reference: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> dict[str, float]:
        if reference.shape != reconstructed.shape:
            raise ValueError(
                "Quantization benchmark tensors must have identical shapes"
            )
        return quantization_error(reference, reconstructed)


def _weight_tensors(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if name and isinstance(weight, torch.Tensor) and weight.ndim >= 2:
            yield name, weight.detach()


@BENCHMARK_REGISTRY.register(
    "model_sparsity",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("compression", "pruning", "weight_sparsity"),
)
class ModelSparsityBenchmark:
    """Measure effective zeros in matrix-like model weights."""

    def __call__(self, model: nn.Module) -> dict[str, Any]:
        module_metrics: dict[str, dict[str, int | float]] = {}
        zero_count = 0
        value_count = 0
        for name, weight in _weight_tensors(model):
            zeros = int(torch.count_nonzero(weight == 0).item())
            values = weight.numel()
            zero_count += zeros
            value_count += values
            module_metrics[name] = {
                "zero_count": zeros,
                "value_count": values,
                "sparsity": zeros / max(values, 1),
            }
        return {
            "zero_count": zero_count,
            "value_count": value_count,
            "sparsity": zero_count / max(value_count, 1),
            "modules": module_metrics,
        }
