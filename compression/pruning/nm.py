"""N:M semi-structured magnitude pruning."""

from __future__ import annotations

import torch

from compression.pruning.base import BasePruner
from compression.pruning.registry import PRUNER_REGISTRY
from core import Maturity, ProjectLevel


@PRUNER_REGISTRY.register(
    "nm",
    "n_m",
    "2:4",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("weight_pruning", "semi_structured", "n:m", "2:4"),
)
class NMPruner(BasePruner):
    """Keep the N largest magnitudes in every contiguous group of M."""

    pattern = "n:m"

    def __init__(self, keep: int = 2, block_size: int = 4, axis: int = -1):
        self.keep = int(keep)
        self.block_size = int(block_size)
        self.axis = int(axis)
        if self.block_size <= 0 or self.keep <= 0 or self.keep >= self.block_size:
            raise ValueError("N:M pruning requires 0 < keep < block_size")

    def compute_mask(self, weight: torch.Tensor) -> torch.Tensor:
        axis = self.axis if self.axis >= 0 else weight.ndim + self.axis
        if axis < 0 or axis >= weight.ndim:
            raise ValueError(f"axis={self.axis} is invalid for weight shape {tuple(weight.shape)}")
        moved = weight.detach().movedim(axis, -1)
        width = moved.shape[-1]
        if width % self.block_size:
            raise ValueError(
                f"N:M pruning requires dimension {width} to be divisible by "
                f"block_size={self.block_size}"
            )
        grouped = moved.reshape(-1, width // self.block_size, self.block_size)
        indices = torch.topk(grouped.abs(), k=self.keep, dim=-1, largest=True).indices
        mask = torch.zeros_like(grouped, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        return mask.reshape(moved.shape).movedim(-1, axis)
