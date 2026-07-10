"""Resolve deterministic runtime state for an experiment."""

from __future__ import annotations

from core import resolve_device, set_seed
from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage


@STAGE_REGISTRY.register(
    "runtime_setup",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "runtime"),
)
class RuntimeSetupStage(ExperimentStage):
    """Set the random seed and resolve the execution device."""

    provides = ("device",)

    def run(self, context: ExperimentContext) -> None:
        runtime_cfg = dict(context.config.get("runtime", {}))
        seed = runtime_cfg.get("seed", context.config.get("seed", 42))
        set_seed(seed)
        device = resolve_device(runtime_cfg.get("device", "auto"))
        context.provide("device", device)
        context.metrics["device"] = str(device)
        context.metrics["seed"] = seed
