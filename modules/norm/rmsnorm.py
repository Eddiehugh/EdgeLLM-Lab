"""RMSNorm."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.norm.registry import NORM_REGISTRY


@NORM_REGISTRY.register("rmsnorm")
class RMSNorm(nn.Module):
    """Root mean square normalization used by LLaMA-like models."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
