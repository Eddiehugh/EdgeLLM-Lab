"""Sparse attention variants."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention.base import merge_heads, split_heads
from modules.attention.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "topk_sparse",
    capabilities=("self_attention", "topk_sparse"),
)
class TopKSparseAttention(nn.Module):
    """Keep only the top-k attention scores for each query."""

    def __init__(self, hidden_size: int, num_heads: int, top_k: int = 64):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.top_k = top_k

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = split_heads(self.wq(x), self.num_heads)
        k = split_heads(self.wk(x), self.num_heads)
        v = split_heads(self.wv(x), self.num_heads)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        k_count = min(self.top_k, scores.size(-1))
        values, indices = torch.topk(scores, k=k_count, dim=-1)
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, indices, values)
        attn = F.softmax(sparse_scores, dim=-1)
        return self.wo(merge_heads(attn @ v))
