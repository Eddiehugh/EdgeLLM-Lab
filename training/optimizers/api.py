"""Public optimizer construction API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from training.optimizers.registry import OPTIMIZER_REGISTRY, PARAM_GROUP_POLICY_REGISTRY


def resolve_optimizer_name(config: str | Mapping[str, Any] | None) -> str:
    """Resolve legacy and structured optimizer configs to a registry name."""

    if config is None:
        return "torch_adamw"
    if isinstance(config, str):
        return config
    if "type" in config:
        return str(config["type"])

    algorithm = str(config.get("algorithm", "adamw"))
    implementation = str(config.get("implementation", "torch"))
    return f"{implementation}_{algorithm}"


def resolve_param_group_policy_name(
    config: str | Mapping[str, Any] | None,
) -> str:
    """Resolve a parameter-group policy config to its registry name."""

    if config is None:
        return "all"
    if isinstance(config, str):
        return config
    return str(config.get("type", "all"))


def _build_parameter_groups(
    policy_config: str | Mapping[str, Any] | None,
    *,
    model: torch.nn.Module | None,
    params: Iterable[torch.nn.Parameter] | None,
) -> list[dict[str, object]]:
    policy_name = resolve_param_group_policy_name(policy_config)
    policy_kwargs: dict[str, Any] = {}
    if isinstance(policy_config, Mapping):
        policy_kwargs = dict(policy_config)
        policy_kwargs.pop("type", None)

    if model is not None:
        named_parameters = list(model.named_parameters())
    elif params is not None:
        if policy_name not in {"all", "default"}:
            raise ValueError(
                "Named parameter-group policies require model= so parameter names "
                "and dimensions remain available"
            )
        named_parameters = [
            (f"parameter_{index}", parameter)
            for index, parameter in enumerate(params)
        ]
    else:
        raise ValueError("build_optimizer requires either model= or params=")

    groups = PARAM_GROUP_POLICY_REGISTRY.build(
        policy_name,
        named_parameters=named_parameters,
        **policy_kwargs,
    )
    if not groups:
        raise ValueError("Optimizer parameter-group policy selected no trainable parameters")
    return groups


def build_optimizer(
    optimizer_config: str | Mapping[str, Any] | None = None,
    *,
    model: torch.nn.Module | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Build an optimizer implementation over independently selected groups."""

    optimizer_name = resolve_optimizer_name(optimizer_config)
    config_kwargs: dict[str, Any] = {}
    policy_config: str | Mapping[str, Any] | None = None
    if isinstance(optimizer_config, Mapping):
        config_kwargs = dict(optimizer_config)
        config_kwargs.pop("type", None)
        config_kwargs.pop("algorithm", None)
        config_kwargs.pop("implementation", None)
        policy_config = config_kwargs.pop("param_group_policy", None)

    parameter_groups = _build_parameter_groups(
        policy_config,
        model=model,
        params=params,
    )
    return OPTIMIZER_REGISTRY.build(
        optimizer_name,
        params=parameter_groups,
        **{**kwargs, **config_kwargs},
    )
