"""Learning-oriented patch Transformer vision encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from core import Maturity, ProjectLevel
from modules.block import build_block
from modules.multimodal import ModalityFeatures
from modules.norm import build_norm
from modules.vision.encoder.registry import VISION_ENCODER_REGISTRY


@VISION_ENCODER_REGISTRY.register(
    "patch_transformer",
    "tiny_vit",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("vision", "patch_embedding", "bidirectional_transformer", "multi_image"),
)
class PatchTransformerVisionEncoder(nn.Module):
    """Encode fixed-size images as patch tokens using reusable Transformer blocks."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        in_channels: int = 3,
        max_images: int = 1,
        block_type: str | dict = "transformer",
        attention_type: str | dict = "mha",
        norm_type: str | dict = "layernorm",
        mlp_type: str | dict = "gelu",
        intermediate_size: int | None = None,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if max_images <= 0:
            raise ValueError("max_images must be positive")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.in_channels = int(in_channels)
        self.max_images = int(max_images)
        self.patches_per_image = (image_size // patch_size) ** 2
        self.max_output_tokens = self.patches_per_image * self.max_images

        self.patch_embedding = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_position = nn.Parameter(
            torch.zeros(1, 1, self.patches_per_image, hidden_size)
        )
        self.image_position = nn.Parameter(
            torch.zeros(1, self.max_images, 1, hidden_size)
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
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        nn.init.trunc_normal_(self.image_position, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> ModalityFeatures:
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width] or "
                "[batch, images, channels, height, width]"
            )
        batch_size, num_images, channels, height, width = pixel_values.shape
        if num_images <= 0:
            raise ValueError("pixel_values must contain at least one image")
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} image channels, received {channels}"
            )
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} images, "
                f"received {height}x{width}"
            )
        if num_images > self.max_images:
            raise ValueError(
                f"Received {num_images} images, max_images={self.max_images}"
            )
        if image_mask is None:
            image_mask = torch.ones(
                (batch_size, num_images),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        elif image_mask.shape != (batch_size, num_images):
            raise ValueError("image_mask must have shape [batch, images]")
        else:
            image_mask = image_mask.to(device=pixel_values.device, dtype=torch.bool)

        images = pixel_values.reshape(
            batch_size * num_images,
            channels,
            height,
            width,
        )
        tokens = self.patch_embedding(images).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(
            batch_size,
            num_images,
            self.patches_per_image,
            self.hidden_size,
        )
        tokens = (
            tokens
            + self.patch_position
            + self.image_position[:, :num_images]
        )
        tokens = tokens.reshape(batch_size, -1, self.hidden_size)
        token_mask = image_mask.repeat_interleave(self.patches_per_image, dim=1)
        attention_token_mask = token_mask.clone()
        empty_samples = ~attention_token_mask.any(dim=1)
        attention_token_mask[empty_samples, 0] = True
        attention_mask = torch.zeros(
            (batch_size, 1, 1, tokens.shape[1]),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        attention_mask = attention_mask.masked_fill(
            ~attention_token_mask[:, None, None, :],
            float("-inf"),
        )
        for block in self.blocks:
            tokens = block(tokens, mask=attention_mask)
        tokens = self.norm(tokens)
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0)
        return ModalityFeatures(tokens, token_mask)
