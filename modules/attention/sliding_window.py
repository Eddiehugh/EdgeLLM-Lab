"""Sliding-window local attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.attention.base import apply_attention, merge_heads, split_heads
from modules.attention.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("sliding_window")
class SlidingWindowAttention(nn.Module):
    """Causal local attention restricted to a fixed window."""

    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 128):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        distance = positions.view(seq_len, 1) - positions.view(1, seq_len)
        blocked = (distance < 0) | (distance >= self.window_size)
        mask = torch.zeros((seq_len, seq_len), device=device)
        return mask.masked_fill(blocked, float("-inf")).view(1, 1, seq_len, seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = split_heads(self.wq(x), self.num_heads)
        k = split_heads(self.wk(x), self.num_heads)
        v = split_heads(self.wv(x), self.num_heads)
        local_mask = self._local_mask(x.size(1), x.device)
        mask = local_mask if mask is None else mask + local_mask
        return self.wo(merge_heads(apply_attention(q, k, v, mask=mask)))
