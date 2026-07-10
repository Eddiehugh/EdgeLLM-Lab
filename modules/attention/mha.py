"""Multi-Head Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.attention.base import apply_attention, merge_heads, split_heads
from modules.attention.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mha")
class MultiHeadAttention(nn.Module):
    """Standard dense Multi-Head Attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = split_heads(self.wq(x), self.num_heads)
        k = split_heads(self.wk(x), self.num_heads)
        v = split_heads(self.wv(x), self.num_heads)
        return self.wo(merge_heads(apply_attention(q, k, v, mask=mask)))
