"""Prefix-token multimodal fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from core import Maturity, ProjectLevel
from modules.multimodal.fusion.registry import MULTIMODAL_FUSION_REGISTRY
from modules.multimodal.types import FusionOutput, ModalityFeatures


@MULTIMODAL_FUSION_REGISTRY.register(
    "prefix",
    "visual_prefix",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("multimodal_fusion", "decoder_prefix", "causal_lm"),
)
class PrefixMultimodalFusion(nn.Module):
    """Place projected modality tokens before text embeddings."""

    def forward(
        self,
        text_embeddings: torch.Tensor,
        modality: ModalityFeatures,
        text_attention_mask: torch.Tensor | None = None,
    ) -> FusionOutput:
        if text_embeddings.ndim != 3:
            raise ValueError("text_embeddings must have shape [batch, tokens, hidden]")
        if text_embeddings.shape[0] != modality.batch_size:
            raise ValueError("Text and modality batch sizes must match")
        if text_embeddings.shape[2] != modality.embeddings.shape[2]:
            raise ValueError("Text and projected modality hidden sizes must match")

        batch_size, text_tokens, _ = text_embeddings.shape
        if text_attention_mask is None:
            text_mask = torch.ones(
                (batch_size, text_tokens),
                dtype=torch.bool,
                device=text_embeddings.device,
            )
        else:
            if text_attention_mask.shape != (batch_size, text_tokens):
                raise ValueError("text_attention_mask shape does not match input_ids")
            text_mask = text_attention_mask.to(
                device=text_embeddings.device,
                dtype=torch.bool,
            )

        embeddings = torch.cat((modality.embeddings, text_embeddings), dim=1)
        attention_mask = torch.cat((modality.valid_mask(), text_mask), dim=1)
        positions = torch.arange(
            modality.token_count,
            modality.token_count + text_tokens,
            device=text_embeddings.device,
        )
        text_positions = positions.unsqueeze(0).expand(batch_size, -1)
        return FusionOutput(embeddings, attention_mask, text_positions)
