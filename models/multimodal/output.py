"""Outputs shared by multimodal causal language models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MultimodalCausalLMOutput:
    """Text logits plus modality accounting for training and benchmarks."""

    logits: torch.Tensor
    modality_token_count: torch.Tensor
    text_hidden_states: torch.Tensor | None = None
    modality_hidden_states: torch.Tensor | None = None
    modality_attention_mask: torch.Tensor | None = None
