"""MLP and FFN modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.registry import Registry, build_from_config


MLP_REGISTRY = Registry[nn.Module]("mlp")


@MLP_REGISTRY.register("gelu")
class GELUMLP(nn.Module):
    """Standard GPT-style MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


@MLP_REGISTRY.register("swiglu")
class SwiGLUMLP(nn.Module):
    """LLaMA-style gated MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def build_mlp(
    mlp_type: str | dict = "gelu",
    *,
    hidden_size: int,
    intermediate_size: int,
    **kwargs,
) -> nn.Module:
    """Build an MLP module by name."""
    return build_from_config(
        MLP_REGISTRY,
        mlp_type,
        default_type="gelu",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        **kwargs,
    )
