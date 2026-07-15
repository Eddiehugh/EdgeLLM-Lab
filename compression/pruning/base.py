"""Core pruning contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch


def validate_sparsity(sparsity: float) -> float:
    value = float(sparsity)
    if not 0.0 <= value < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    return value


class BasePruner(ABC):
    """Compute deterministic binary masks without owning model traversal."""

    pattern = "unstructured"

    @abstractmethod
    def compute_mask(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_masks(
        self, weights: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {name: self.compute_mask(weight) for name, weight in weights.items()}
