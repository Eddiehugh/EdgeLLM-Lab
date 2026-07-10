"""Learning-oriented Multi-head Latent Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.attention.base import apply_attention, merge_heads
from modules.attention.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "mla",
    capabilities=("self_attention", "latent_kv", "learning_reference"),
)
class MultiHeadLatentAttention(nn.Module):
    """Simplified MLA-style attention with low-rank latent KV states.

    This is not a byte-for-byte DeepSeek MLA implementation. It is a compact
    learning version that makes the latent KV idea experimentable.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        latent_size: int | None = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        latent_size = latent_size or max(self.head_dim, hidden_size // 4)

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.latent_down = nn.Linear(hidden_size, latent_size, bias=False)
        self.latent_to_k = nn.Linear(latent_size, hidden_size, bias=False)
        self.latent_to_v = nn.Linear(latent_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        latent = self.latent_down(x)
        k = self.latent_to_k(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.latent_to_v(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return self.wo(merge_heads(apply_attention(q, k, v, mask=mask)))
