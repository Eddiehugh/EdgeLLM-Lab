"""Shared tensor contracts for multimodal components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModalityFeatures:
    """Modality token embeddings and their valid-token mask."""

    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("Modality embeddings must have shape [batch, tokens, hidden]")
        if self.attention_mask is not None:
            expected = self.embeddings.shape[:2]
            if self.attention_mask.shape != expected:
                raise ValueError(
                    "Modality attention mask must have shape "
                    f"{tuple(expected)}, received {tuple(self.attention_mask.shape)}"
                )

    @property
    def batch_size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def token_count(self) -> int:
        return self.embeddings.shape[1]

    def valid_mask(self) -> torch.Tensor:
        if self.attention_mask is None:
            return torch.ones(
                self.embeddings.shape[:2],
                dtype=torch.bool,
                device=self.embeddings.device,
            )
        return self.attention_mask.to(device=self.embeddings.device, dtype=torch.bool)

    def with_embeddings(self, embeddings: torch.Tensor) -> "ModalityFeatures":
        return ModalityFeatures(embeddings, self.attention_mask)


@dataclass(frozen=True)
class FusionOutput:
    """Fused decoder inputs plus positions corresponding to text logits."""

    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    text_positions: torch.Tensor

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("Fused embeddings must have shape [batch, tokens, hidden]")
        if self.attention_mask.shape != self.embeddings.shape[:2]:
            raise ValueError("Fused attention mask shape does not match embeddings")
        if self.text_positions.ndim != 2:
            raise ValueError("text_positions must have shape [batch, text_tokens]")
        if self.text_positions.shape[0] != self.embeddings.shape[0]:
            raise ValueError("text_positions batch size does not match embeddings")
