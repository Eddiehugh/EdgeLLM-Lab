"""Normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.registry import Registry, build_from_config


NORM_REGISTRY = Registry[nn.Module]("norm")


@NORM_REGISTRY.register("layernorm")
class LayerNorm(nn.LayerNorm):
    """LayerNorm with the same hidden_size signature as RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps=eps)


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


def build_norm(
    norm_type: str | dict = "layernorm",
    *,
    hidden_size: int,
    **kwargs,
) -> nn.Module:
    """Build a normalization module by name."""
    return build_from_config(
        NORM_REGISTRY,
        norm_type,
        default_type="layernorm",
        hidden_size=hidden_size,
        **kwargs,
    )
