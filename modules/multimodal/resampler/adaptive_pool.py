"""Adaptive average-pooling modality token resampler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import Maturity, ProjectLevel
from modules.multimodal.resampler.registry import MULTIMODAL_RESAMPLER_REGISTRY
from modules.multimodal.types import ModalityFeatures


@MULTIMODAL_RESAMPLER_REGISTRY.register(
    "adaptive_pool",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("token_resampling", "fixed_budget", "mask_aware"),
)
class AdaptivePoolMultimodalResampler(nn.Module):
    """Compress modality tokens to a fixed budget with masked average pooling."""

    def __init__(self, num_tokens: int):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        self.num_tokens = int(num_tokens)

    def output_token_count(self, input_tokens: int) -> int:
        if self.num_tokens > input_tokens:
            raise ValueError(
                "adaptive_pool cannot expand the modality token sequence: "
                f"{self.num_tokens} > {input_tokens}"
            )
        return self.num_tokens

    def forward(self, features: ModalityFeatures) -> ModalityFeatures:
        embeddings = features.embeddings
        if not embeddings.is_floating_point():
            raise TypeError("adaptive_pool requires floating-point embeddings")
        if self.num_tokens > embeddings.shape[1]:
            raise ValueError("adaptive_pool num_tokens exceeds the input token count")
        mask = features.valid_mask()
        weights = mask.to(dtype=embeddings.dtype).unsqueeze(1)
        values = embeddings.transpose(1, 2) * weights
        pooled_values = F.adaptive_avg_pool1d(values, self.num_tokens)
        pooled_weights = F.adaptive_avg_pool1d(weights, self.num_tokens)
        pooled = pooled_values / pooled_weights.clamp_min(torch.finfo(embeddings.dtype).eps)
        pooled = pooled.transpose(1, 2)
        pooled_mask = pooled_weights.squeeze(1) > 0
        pooled = pooled.masked_fill(~pooled_mask.unsqueeze(-1), 0)
        return ModalityFeatures(pooled, pooled_mask)
