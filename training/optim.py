"""Replaceable optimizer factories."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from core.registry import Registry, build_from_config


OPTIMIZER_REGISTRY = Registry("optimizer")


@OPTIMIZER_REGISTRY.register("adamw")
def build_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    **kwargs,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        **kwargs,
    )


@OPTIMIZER_REGISTRY.register("sgd")
def build_sgd(
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-2,
    momentum: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=lr, momentum=momentum, **kwargs)


def build_optimizer(
    optimizer_type: str | dict = "adamw",
    *,
    params: Iterable[torch.nn.Parameter],
    **kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer by name."""
    return build_from_config(
        OPTIMIZER_REGISTRY,
        optimizer_type,
        default_type="adamw",
        params=params,
        **kwargs,
    )
