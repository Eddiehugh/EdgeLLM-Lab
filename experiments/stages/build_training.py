"""Build loss, optimizer, and scheduler components."""

from __future__ import annotations

from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage
from experiments.stages.common import selected_name
from training import (
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    build_loss,
    build_optimizer,
    build_scheduler,
)


@STAGE_REGISTRY.register(
    "build_training",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "training"),
)
class BuildTrainingStage(ExperimentStage):
    """Build the loss, optimizer, and learning-rate scheduler."""

    requires = ("model",)
    provides = ("loss", "optimizer", "scheduler")

    def run(self, context: ExperimentContext) -> None:
        model = context.require("model")
        training_cfg = dict(context.config.get("training", {}))

        loss_cfg = context.config.get("loss", "causal_lm")
        context.provide("loss", build_loss(loss_cfg))
        context.track_component(
            "loss", LOSS_REGISTRY, selected_name(loss_cfg, "causal_lm")
        )

        optimizer_cfg = training_cfg.get("optimizer", {"type": "adamw"})
        if isinstance(optimizer_cfg, dict) and "lr" not in optimizer_cfg:
            optimizer_cfg = dict(optimizer_cfg)
            optimizer_cfg["lr"] = training_cfg.get("learning_rate", 3e-4)
        optimizer = build_optimizer(optimizer_cfg, params=model.parameters())
        context.provide("optimizer", optimizer)
        context.track_component(
            "optimizer", OPTIMIZER_REGISTRY, selected_name(optimizer_cfg, "adamw")
        )

        scheduler_cfg = training_cfg.get("scheduler", {"type": "constant"})
        if isinstance(scheduler_cfg, dict) and "max_steps" not in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            scheduler_cfg["max_steps"] = training_cfg.get("max_steps", 1)
        scheduler = build_scheduler(scheduler_cfg, optimizer=optimizer)
        context.provide("scheduler", scheduler)
        context.track_component(
            "scheduler", SCHEDULER_REGISTRY, selected_name(scheduler_cfg, "constant")
        )
