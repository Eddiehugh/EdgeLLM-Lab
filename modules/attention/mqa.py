"""Multi-Query Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.attention.base import apply_attention, merge_heads
from modules.attention.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "mqa",
    capabilities=("self_attention", "dense", "shared_kv"),
)
class MultiQueryAttention(nn.Module):
    """MHA queries with a single shared key/value head."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, 1, seq_len, self.head_dim)
        v = self.wv(x).view(batch_size, 1, seq_len, self.head_dim)
        k = k.expand(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.expand(batch_size, self.num_heads, seq_len, self.head_dim)
        return self.wo(merge_heads(apply_attention(q, k, v, mask=mask)))
