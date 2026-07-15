"""Two-layer MLP modality projector."""

from __future__ import annotations

import torch
import torch.nn as nn

from core import Maturity, ProjectLevel
from modules.multimodal.projector.registry import MULTIMODAL_PROJECTOR_REGISTRY


@MULTIMODAL_PROJECTOR_REGISTRY.register(
    "mlp",
    "mlp2x_gelu",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("modality_projection", "nonlinear", "gelu"),
)
class MLPMultimodalProjector(nn.Module):
    """Project modality tokens through a GELU MLP."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        hidden_size = hidden_size or output_size
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=bias),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)
