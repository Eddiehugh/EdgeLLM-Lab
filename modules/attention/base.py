"""Shared helpers for attention implementations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, hidden_size = x.shape
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    head_dim = hidden_size // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)


def apply_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def repeat_kv(x: torch.Tensor, repeat_factor: int) -> torch.Tensor:
    if repeat_factor == 1:
        return x
    return x.repeat_interleave(repeat_factor, dim=1)
