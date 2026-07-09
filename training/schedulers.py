"""Replaceable learning-rate scheduler factories."""

from __future__ import annotations

import torch

from core.registry import Registry, build_from_config


SCHEDULER_REGISTRY = Registry("scheduler")


@SCHEDULER_REGISTRY.register("constant")
def build_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    **_: object,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


@SCHEDULER_REGISTRY.register("cosine")
def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    min_lr_ratio: float = 0.1,
    **_: object,
) -> torch.optim.lr_scheduler.LRScheduler:
    def lr_lambda(step: int) -> float:
        progress = min(step / max(max_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    scheduler_type: str | dict = "constant",
    *,
    optimizer: torch.optim.Optimizer,
    **kwargs,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a learning-rate scheduler by name."""
    return build_from_config(
        SCHEDULER_REGISTRY,
        scheduler_type,
        default_type="constant",
        optimizer=optimizer,
        **kwargs,
    )
