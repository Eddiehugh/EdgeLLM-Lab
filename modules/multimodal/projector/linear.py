"""Linear modality-to-language projector."""

from __future__ import annotations

import torch
import torch.nn as nn

from core import Maturity, ProjectLevel
from modules.multimodal.projector.registry import MULTIMODAL_PROJECTOR_REGISTRY


@MULTIMODAL_PROJECTOR_REGISTRY.register(
    "linear",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("modality_projection", "single_linear"),
)
class LinearMultimodalProjector(nn.Module):
    """Map modality features directly into the language hidden space."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.projection = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)
