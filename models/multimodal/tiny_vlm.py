"""Tiny decoder-only vision-language model for architecture experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from core import Maturity, ProjectLevel
from models.multimodal.output import MultimodalCausalLMOutput
from models.registry import MODEL_REGISTRY
from modules.block import build_block
from modules.multimodal import (
    ModalityFeatures,
    build_multimodal_fusion,
    build_multimodal_projector,
    build_multimodal_resampler,
)
from modules.norm import build_norm
from modules.vision import build_vision_encoder


def _selected_name(config: str | Mapping[str, Any], default: str) -> str:
    if isinstance(config, str):
        return config
    return str(config.get("type", default))


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


@MODEL_REGISTRY.register(
    "tiny_vlm",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=(
        "causal_lm",
        "multimodal",
        "vision_language",
        "training",
        "modality_scopes",
    ),
)
class TinyVisionLanguageModel(nn.Module):
    """Compose a vision tower, resampler, projector, and decoder-only LM."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_position_embeddings: int,
        image_size: int,
        patch_size: int,
        vision_hidden_size: int,
        vision_num_layers: int = 1,
        vision_num_heads: int = 4,
        in_channels: int = 3,
        max_images: int = 1,
        vision_encoder_type: str | dict = "patch_transformer",
        projector_type: str | dict = "linear",
        resampler_type: str | dict = "identity",
        fusion_type: str | dict = "prefix",
        block_type: str | dict = "transformer",
        attention_type: str | dict = "mha",
        norm_type: str | dict = "layernorm",
        mlp_type: str | dict = "gelu",
        vision_block_type: str | dict = "transformer",
        vision_attention_type: str | dict = "mha",
        vision_norm_type: str | dict = "layernorm",
        vision_mlp_type: str | dict = "gelu",
        intermediate_size: int | None = None,
        vision_intermediate_size: int | None = None,
        tie_word_embeddings: bool = False,
        freeze_vision_encoder: bool = False,
        freeze_language_model: bool = False,
        freeze_projector: bool = False,
    ):
        super().__init__()
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        self.max_position_embeddings = int(max_position_embeddings)

        self.vision_encoder = build_vision_encoder(
            vision_encoder_type,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=vision_hidden_size,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            in_channels=in_channels,
            max_images=max_images,
            block_type=vision_block_type,
            attention_type=vision_attention_type,
            norm_type=vision_norm_type,
            mlp_type=vision_mlp_type,
            intermediate_size=vision_intermediate_size,
        )
        self.resampler = build_multimodal_resampler(resampler_type)
        self.multimodal_projector = build_multimodal_projector(
            projector_type,
            input_size=vision_hidden_size,
            output_size=hidden_size,
        )
        self.fusion = build_multimodal_fusion(fusion_type)

        encoder_tokens = getattr(self.vision_encoder, "max_output_tokens", None)
        if not isinstance(encoder_tokens, int) or encoder_tokens <= 0:
            raise TypeError("Vision encoder must expose a positive max_output_tokens")
        output_token_count = getattr(self.resampler, "output_token_count", None)
        if not callable(output_token_count):
            raise TypeError("Multimodal resampler must implement output_token_count")
        self.max_visual_tokens = int(output_token_count(encoder_tokens))

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(
            self.max_position_embeddings + self.max_visual_tokens,
            hidden_size,
        )
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
            self.lm_head.weight = self.token_embedding.weight

        self._component_selections = {
            "vision_encoder": _selected_name(
                vision_encoder_type,
                "patch_transformer",
            ),
            "multimodal_projector": _selected_name(projector_type, "linear"),
            "multimodal_resampler": _selected_name(resampler_type, "identity"),
            "multimodal_fusion": _selected_name(fusion_type, "prefix"),
            "vision_block": _selected_name(vision_block_type, "transformer"),
            "vision_attention": _selected_name(vision_attention_type, "mha"),
            "vision_norm": _selected_name(vision_norm_type, "layernorm"),
            "vision_mlp": _selected_name(vision_mlp_type, "gelu"),
        }

        if freeze_vision_encoder:
            _set_trainable(self.vision_encoder, False)
        if freeze_projector:
            _set_trainable(self.multimodal_projector, False)
        if freeze_language_model:
            for module in (
                self.token_embedding,
                self.position_embedding,
                self.blocks,
                self.norm,
                self.lm_head,
            ):
                _set_trainable(module, False)

    def component_selections(self) -> dict[str, str]:
        return dict(self._component_selections)

    @staticmethod
    def compression_scopes() -> dict[str, tuple[str, ...]]:
        return {
            "language": (
                "token_embedding",
                "position_embedding",
                "blocks",
                "norm",
                "lm_head",
            ),
            "vision": ("vision_encoder",),
            "projector": ("multimodal_projector",),
            "resampler": ("resampler",),
            "fusion": ("fusion",),
            "multimodal": (
                "vision_encoder",
                "resampler",
                "multimodal_projector",
                "fusion",
            ),
        }

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> ModalityFeatures:
        features = self.vision_encoder(pixel_values, image_mask=image_mask)
        features = self.resampler(features)
        projected = self.multimodal_projector(features.embeddings)
        return features.with_embeddings(projected)

    def _decoder_mask(
        self,
        valid_tokens: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sequence_length = valid_tokens.shape[1]
        causal = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                dtype=dtype,
                device=valid_tokens.device,
            ),
            diagonal=1,
        ).view(1, 1, sequence_length, sequence_length)
        key_padding = torch.zeros(
            (valid_tokens.shape[0], 1, 1, sequence_length),
            dtype=dtype,
            device=valid_tokens.device,
        ).masked_fill(~valid_tokens[:, None, None, :], float("-inf"))
        return causal + key_padding

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> MultimodalCausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, text_tokens]")
        if input_ids.shape[1] > self.max_position_embeddings:
            raise ValueError(
                f"text length {input_ids.shape[1]} exceeds "
                f"max_position_embeddings={self.max_position_embeddings}"
            )

        modality = self.encode_images(pixel_values, image_mask=image_mask)
        text_embeddings = self.token_embedding(input_ids)
        fused = self.fusion(
            text_embeddings,
            modality,
            text_attention_mask=attention_mask,
        )
        if fused.embeddings.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError("Fused multimodal sequence exceeds the position table")

        positions = torch.arange(
            fused.embeddings.shape[1],
            device=fused.embeddings.device,
        ).unsqueeze(0)
        hidden = fused.embeddings + self.position_embedding(positions)
        decoder_mask = self._decoder_mask(fused.attention_mask, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask=decoder_mask)
        hidden = self.norm(hidden)

        gather_index = fused.text_positions.unsqueeze(-1).expand(
            -1,
            -1,
            hidden.shape[-1],
        )
        text_hidden = hidden.gather(1, gather_index)
        logits = self.lm_head(text_hidden)
        modality_token_count = modality.valid_mask().sum(dim=1)
        return MultimodalCausalLMOutput(
            logits=logits,
            modality_token_count=modality_token_count,
            text_hidden_states=text_hidden,
            modality_hidden_states=modality.embeddings,
            modality_attention_mask=modality.valid_mask(),
        )
