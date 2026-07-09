"""Attention zoo.

This file will gradually include:
- Multi-Head Attention
- Multi-Query Attention
- Grouped-Query Attention
- RoPE Attention
- Sliding Window Attention
- Multi-head Latent Attention
- Sparse Attention
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import Registry, build_from_config


ATTENTION_REGISTRY = Registry[nn.Module]("attention")


@ATTENTION_REGISTRY.register("mha")
class MultiHeadAttention(nn.Module):
    """Minimal Multi-Head Attention implementation for TinyGPT v0.1."""

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
        batch_size, seq_len, hidden_size = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.wo(out)


def build_attention(
    attention_type: str | dict = "mha",
    *,
    hidden_size: int,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """Build an Attention module by name."""
    return build_from_config(
        ATTENTION_REGISTRY,
        attention_type,
        default_type="mha",
        hidden_size=hidden_size,
        num_heads=num_heads,
        **kwargs,
    )
