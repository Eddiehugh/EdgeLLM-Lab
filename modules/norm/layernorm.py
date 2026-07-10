"""LayerNorm."""

from __future__ import annotations

import torch.nn as nn

from modules.norm.registry import NORM_REGISTRY


@NORM_REGISTRY.register("layernorm", capabilities=("normalization", "mean_centered"))
class LayerNorm(nn.LayerNorm):
    """LayerNorm with the same hidden_size signature as RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps=eps)
