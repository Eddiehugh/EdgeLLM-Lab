"""Adapters for mature optimizer implementations provided by PyTorch."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from core import Maturity, ProjectLevel
from training.optimizers.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register(
    "torch_adamw",
    "adamw",
    level=ProjectLevel.WORK,
    maturity=Maturity.PRODUCTION,
    capabilities=("gradient_optimization", "adamw", "torch_backend"),
)
def build_torch_adamw(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, object]],
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    **kwargs: object,
) -> torch.optim.Optimizer:
    """Build the production PyTorch AdamW implementation."""

    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        **kwargs,
    )


@OPTIMIZER_REGISTRY.register(
    "torch_sgd",
    "sgd",
    level=ProjectLevel.WORK,
    maturity=Maturity.PRODUCTION,
    capabilities=("gradient_optimization", "sgd", "torch_backend"),
)
def build_torch_sgd(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, object]],
    lr: float = 1e-2,
    momentum: float = 0.0,
    **kwargs: object,
) -> torch.optim.Optimizer:
    """Build the production PyTorch SGD implementation."""

    return torch.optim.SGD(params, lr=lr, momentum=momentum, **kwargs)
