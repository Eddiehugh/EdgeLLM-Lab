"""MoE routing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.moe.registry import MOE_REGISTRY


@MOE_REGISTRY.register("topk_router", capabilities=("routing", "top_k"))
class TopKRouter(nn.Module):
    """Top-k expert router used as a lightweight MoE building block."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        weights, indices = torch.topk(logits, k=self.top_k, dim=-1)
        return torch.softmax(weights, dim=-1), indices
