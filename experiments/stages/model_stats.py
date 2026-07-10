"""Collect architecture-independent model statistics."""

from __future__ import annotations

from core import count_parameters, model_size_bytes
from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage


@STAGE_REGISTRY.register(
    "model_stats",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "metrics"),
)
class ModelStatsStage(ExperimentStage):
    """Collect parameter counts and in-memory model size."""

    requires = ("model",)

    def run(self, context: ExperimentContext) -> None:
        model = context.require("model")
        context.metrics.update(
            {
                "parameter_count": count_parameters(model),
                "trainable_parameter_count": count_parameters(
                    model, trainable_only=True
                ),
                "model_size_mb": model_size_bytes(model) / (1024 * 1024),
            }
        )
