"""Persist the model when checkpointing is enabled."""

from __future__ import annotations

from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage


@STAGE_REGISTRY.register(
    "checkpoint",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.EXPERIMENTAL,
    capabilities=("pipeline", "artifact"),
)
class CheckpointStage(ExperimentStage):
    """Save an optional model checkpoint as a tracked artifact."""

    requires = ("model",)

    def run(self, context: ExperimentContext) -> None:
        training_cfg = dict(context.config.get("training", {}))
        if not bool(training_cfg.get("save_checkpoint", False)):
            return
        checkpoint_path = context.run.write_checkpoint(context.require("model"))
        context.record_artifact("checkpoint", checkpoint_path)
        context.metrics["checkpoint_path"] = str(checkpoint_path)
