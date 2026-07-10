"""Pre-norm Transformer block."""

from __future__ import annotations

import torch.nn as nn

from modules.attention import build_attention
from modules.block.registry import BLOCK_REGISTRY
from modules.mlp import build_mlp
from modules.norm import build_norm


@BLOCK_REGISTRY.register(
    "transformer",
    capabilities=("pre_norm", "residual", "self_attention", "mlp"),
)
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
