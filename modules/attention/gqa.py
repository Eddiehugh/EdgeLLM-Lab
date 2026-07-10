"""Grouped-Query Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.attention.base import apply_attention, merge_heads, repeat_kv
from modules.attention.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("gqa")
class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with fewer KV heads than Q heads."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        num_kv_heads = num_kv_heads or max(1, num_heads // 2)
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.repeat_factor = num_heads // num_kv_heads

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = repeat_kv(k, self.repeat_factor)
        v = repeat_kv(v, self.repeat_factor)
        return self.wo(merge_heads(apply_attention(q, k, v, mask=mask)))
