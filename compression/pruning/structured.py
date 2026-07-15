"""Structured channel masking by vector norm."""

from __future__ import annotations

import torch

from compression.pruning.base import BasePruner, validate_sparsity
from compression.pruning.registry import PRUNER_REGISTRY
from core import Maturity, ProjectLevel


@PRUNER_REGISTRY.register(
    "channel",
    "structured_channel",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("weight_pruning", "structured", "channel"),
)
class StructuredChannelPruner(BasePruner):
    """Mask complete input or output channels using their Lp norm."""

    pattern = "structured_channel"

    def __init__(self, sparsity: float = 0.5, axis: int = 0, norm: float = 2.0):
        self.sparsity = validate_sparsity(sparsity)
        self.axis = int(axis)
        self.norm = float(norm)
        if self.norm <= 0:
            raise ValueError("norm must be positive")

    def compute_mask(self, weight: torch.Tensor) -> torch.Tensor:
        axis = self.axis if self.axis >= 0 else weight.ndim + self.axis
        if axis < 0 or axis >= weight.ndim:
            raise ValueError(f"axis={self.axis} is invalid for weight shape {tuple(weight.shape)}")
        reduce_dims = tuple(index for index in range(weight.ndim) if index != axis)
        scores = torch.linalg.vector_norm(weight.detach().float(), ord=self.norm, dim=reduce_dims)
        prune_count = int(scores.numel() * self.sparsity)
        keep = torch.ones(scores.numel(), dtype=torch.bool, device=weight.device)
        if prune_count:
            keep[torch.argsort(scores, stable=True)[:prune_count]] = False
        view_shape = [1] * weight.ndim
        view_shape[axis] = weight.shape[axis]
        return keep.view(view_shape).expand_as(weight)
