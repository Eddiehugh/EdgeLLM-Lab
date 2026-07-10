"""Experiment configuration normalization and validation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


CONFIG_SCHEMA_VERSION = 1
DEFAULT_PIPELINE_STAGES = (
    "runtime_setup",
    "build_data",
    "build_model",
    "build_training",
    "train",
    "model_stats",
    "checkpoint",
)


class ExperimentConfigError(ValueError):
    """Raised when an experiment cannot be constructed safely."""


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_stage(stage: Any, index: int, issues: list[str]) -> None:
    if isinstance(stage, str) and stage.strip():
        return
    if isinstance(stage, Mapping):
        name = stage.get("type", stage.get("name"))
        if isinstance(name, str) and name.strip():
            return
    issues.append(
        f"pipeline.stages[{index}] must be a stage name or a mapping with type/name"
    )


def normalize_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated, fully resolved experiment configuration."""

    if not isinstance(config, Mapping):
        raise ExperimentConfigError("Experiment config root must be a mapping")

    normalized = copy.deepcopy(dict(config))
    normalized.setdefault("schema_version", CONFIG_SCHEMA_VERSION)

    pipeline = normalized.get("pipeline")
    if pipeline is None:
        normalized["pipeline"] = {"stages": list(DEFAULT_PIPELINE_STAGES)}
    elif isinstance(pipeline, list):
        normalized["pipeline"] = {"stages": copy.deepcopy(pipeline)}
    elif isinstance(pipeline, Mapping):
        pipeline_cfg = copy.deepcopy(dict(pipeline))
        pipeline_cfg.setdefault("stages", list(DEFAULT_PIPELINE_STAGES))
        normalized["pipeline"] = pipeline_cfg

    validate_experiment_config(normalized)
    return normalized


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate framework-level invariants while allowing extension fields."""

    issues: list[str] = []
    version = config.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        issues.append(
            f"schema_version must be {CONFIG_SCHEMA_VERSION}; received {version!r}"
        )

    for section in ("experiment", "runtime", "model", "data", "loss", "training"):
        value = config.get(section)
        if value is not None and section != "loss" and not isinstance(value, Mapping):
            issues.append(f"{section} must be a mapping")

    loss_cfg = config.get("loss")
    if loss_cfg is not None and not isinstance(loss_cfg, (str, Mapping)):
        issues.append("loss must be a component name or mapping")

    model_cfg = config.get("model", {})
    if isinstance(model_cfg, Mapping):
        for key in (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "max_position_embeddings",
        ):
            if key in model_cfg and not _is_positive_int(model_cfg[key]):
                issues.append(f"model.{key} must be a positive integer")

    data_cfg = config.get("data", {})
    if isinstance(data_cfg, Mapping):
        if "block_size" in data_cfg and not _is_positive_int(data_cfg["block_size"]):
            issues.append("data.block_size must be a positive integer")

    training_cfg = config.get("training", {})
    if isinstance(training_cfg, Mapping):
        for key in ("batch_size", "max_steps", "gradient_accumulation_steps"):
            if key in training_cfg and not _is_positive_int(training_cfg[key]):
                issues.append(f"training.{key} must be a positive integer")

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, Mapping):
        issues.append("pipeline must be a mapping or list")
    else:
        stages = pipeline.get("stages")
        if not isinstance(stages, list) or not stages:
            issues.append("pipeline.stages must be a non-empty list")
        else:
            for index, stage in enumerate(stages):
                _validate_stage(stage, index, issues)

    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise ExperimentConfigError(f"Invalid experiment config:\n{details}")
