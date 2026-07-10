"""Benchmark registry and factory."""

from __future__ import annotations

from typing import Any

import torch

from core import Maturity, ProjectLevel, count_parameters, model_size_bytes
from core.registry import Registry, build_from_config


BENCHMARK_REGISTRY = Registry("benchmark")


@BENCHMARK_REGISTRY.register(
    "model_stats",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("model_size", "parameter_count"),
)
class ModelStatsBenchmark:
    """Collect architecture-independent model size metrics."""

    def __call__(self, model: torch.nn.Module) -> dict[str, Any]:
        return {
            "parameter_count": count_parameters(model),
            "trainable_parameter_count": count_parameters(model, trainable_only=True),
            "model_size_mb": model_size_bytes(model) / (1024 * 1024),
        }


def build_benchmark(benchmark_type: str | dict = "model_stats", **kwargs):
    """Build a benchmark by name."""
    return build_from_config(
        BENCHMARK_REGISTRY,
        benchmark_type,
        default_type="model_stats",
        **kwargs,
    )
