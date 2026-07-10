"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.mlp.registry import MLP_REGISTRY


@MLP_REGISTRY.register("swiglu", capabilities=("feed_forward", "gated", "swiglu"))
class SwiGLUMLP(nn.Module):
    """LLaMA-style gated MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
