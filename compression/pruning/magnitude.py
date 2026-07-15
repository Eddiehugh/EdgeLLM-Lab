"""Local and global unstructured magnitude pruning."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from compression.pruning.base import BasePruner, validate_sparsity
from compression.pruning.registry import PRUNER_REGISTRY
from core import Maturity, ProjectLevel


def _smallest_mask(values: torch.Tensor, prune_count: int) -> torch.Tensor:
    flat = values.detach().abs().reshape(-1)
    mask = torch.ones(flat.numel(), dtype=torch.bool, device=flat.device)
    if prune_count:
        indices = torch.argsort(flat, stable=True)[:prune_count]
        mask[indices] = False
    return mask.reshape(values.shape)


@PRUNER_REGISTRY.register(
    "magnitude",
    "unstructured_magnitude",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("weight_pruning", "unstructured", "local", "global"),
)
class MagnitudePruner(BasePruner):
    """Prune the smallest absolute weights at layer or model scope."""

    pattern = "unstructured"

    def __init__(self, sparsity: float = 0.5, scope: str = "layer"):
        self.sparsity = validate_sparsity(sparsity)
        if scope not in {"layer", "global"}:
            raise ValueError("magnitude pruning scope must be 'layer' or 'global'")
        self.scope = scope

    def compute_mask(self, weight: torch.Tensor) -> torch.Tensor:
        return _smallest_mask(weight, int(weight.numel() * self.sparsity))

    def compute_masks(
        self, weights: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.scope == "layer":
            return super().compute_masks(weights)
        names = list(weights)
        if not names:
            return {}
        flattened = torch.cat([weights[name].detach().abs().reshape(-1) for name in names])
        global_mask = _smallest_mask(
            flattened, int(flattened.numel() * self.sparsity)
        ).reshape(-1)
        masks = {}
        offset = 0
        for name in names:
            count = weights[name].numel()
            masks[name] = global_mask[offset : offset + count].reshape(weights[name].shape)
            offset += count
        return masks
