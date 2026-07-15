"""Identity modality token resampler."""

from __future__ import annotations

import torch.nn as nn

from core import Maturity, ProjectLevel
from modules.multimodal.resampler.registry import MULTIMODAL_RESAMPLER_REGISTRY
from modules.multimodal.types import ModalityFeatures


@MULTIMODAL_RESAMPLER_REGISTRY.register(
    "identity",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("token_resampling", "identity", "mask_preserving"),
)
class IdentityMultimodalResampler(nn.Module):
    """Preserve every modality token and its mask."""

    def output_token_count(self, input_tokens: int) -> int:
        return input_tokens

    def forward(self, features: ModalityFeatures) -> ModalityFeatures:
        return features
