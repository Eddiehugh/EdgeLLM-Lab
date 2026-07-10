"""GELU feed-forward network."""

from __future__ import annotations

import torch.nn as nn

from modules.mlp.registry import MLP_REGISTRY


@MLP_REGISTRY.register("gelu", capabilities=("feed_forward", "gelu"))
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
