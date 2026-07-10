"""Rotary Position Embedding."""

from __future__ import annotations

import torch

from modules.position.registry import POSITION_ENCODING_REGISTRY


@POSITION_ENCODING_REGISTRY.register("rope")
class RotaryPositionEmbedding:
    """Minimal RoPE implementation for query/key tensors."""

    def __init__(self, dim: int, base: float = 10000.0):
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        self.dim = dim
        self.base = base

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(-2)
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype)
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (
            self._rotate_half(k) * sin
        )


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: float = 10000.0):
    """Apply RoPE to query and key tensors."""

    return RotaryPositionEmbedding(q.size(-1), base=base).apply(q, k)
