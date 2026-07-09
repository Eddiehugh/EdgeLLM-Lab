"""TinyGPT model.

This is the first trainable model target of EdgeLLM-Lab.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import MODEL_REGISTRY
from modules.block import build_block
from modules.norm import build_norm


@MODEL_REGISTRY.register("tiny_gpt")
class TinyGPT(nn.Module):
    """A minimal GPT-style causal language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_position_embeddings: int,
        attention_type: str = "mha",
        block_type: str = "transformer",
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
        intermediate_size: int | None = None,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_position_embeddings, hidden_size)
        self.blocks = nn.ModuleList(
            [
                build_block(
                    block_type,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    attention_type=attention_type,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = build_norm(norm_type, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        x = self.norm(x)
        return self.lm_head(x)
