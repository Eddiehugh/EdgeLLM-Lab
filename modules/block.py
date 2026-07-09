"""Transformer block definitions."""

from __future__ import annotations

import torch.nn as nn

from core.registry import Registry, build_from_config
from modules.attention import build_attention
from modules.mlp import build_mlp
from modules.norm import build_norm


BLOCK_REGISTRY = Registry[nn.Module]("block")


@BLOCK_REGISTRY.register("transformer")
class TransformerBlock(nn.Module):
    """Configurable pre-norm Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_type: str | dict = "mha",
        norm_type: str | dict = "layernorm",
        mlp_type: str | dict = "gelu",
        intermediate_size: int | None = None,
    ):
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size

        self.norm1 = build_norm(norm_type, hidden_size=hidden_size)
        self.attn = build_attention(
            attention_type,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.norm2 = build_norm(norm_type, hidden_size=hidden_size)
        self.mlp = build_mlp(
            mlp_type,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


def build_block(
    block_type: str | dict = "transformer",
    *,
    hidden_size: int,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """Build a Transformer block by name."""
    return build_from_config(
        BLOCK_REGISTRY,
        block_type,
        default_type="transformer",
        hidden_size=hidden_size,
        num_heads=num_heads,
        **kwargs,
    )
