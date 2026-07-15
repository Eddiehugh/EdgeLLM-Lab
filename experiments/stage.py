"""Replaceable experiment stage contract and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from core.registry import Registry
from experiments.context import ExperimentContext


STAGE_REGISTRY = Registry[type["ExperimentStage"]]("experiment_stage")
_BUILTINS_LOADED = False


class ExperimentStage(ABC):
    """A small pipeline unit that declares dependencies through the context."""

    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(self, **config: Any):
        self.config = config

    @abstractmethod
    def run(self, context: ExperimentContext) -> None:
        """Execute this stage and update the shared context."""


def load_builtin_stages() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import experiments.stages.build_data  # noqa: F401
    import experiments.stages.build_model  # noqa: F401
    import experiments.stages.build_training  # noqa: F401
    import experiments.stages.checkpoint  # noqa: F401
    import experiments.stages.model_stats  # noqa: F401
    import experiments.stages.prune_model  # noqa: F401
    import experiments.stages.quantize_model  # noqa: F401
    import experiments.stages.runtime_setup  # noqa: F401
    import experiments.stages.train  # noqa: F401


def stage_selector(config: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(config, str):
        return config, {}
    kwargs = dict(config)
    name = kwargs.pop("type", kwargs.pop("name", None))
    if name is None:
        raise KeyError("Experiment stage config requires 'type' or 'name'")
    return str(name), kwargs


def build_stage(config: str | Mapping[str, Any]) -> tuple[str, ExperimentStage]:
    load_builtin_stages()
    name, kwargs = stage_selector(config)
    return name, STAGE_REGISTRY.build(name, **kwargs)
