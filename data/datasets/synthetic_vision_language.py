"""Deterministic synthetic vision-language training data."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from core import Maturity, ProjectLevel
from data.dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register(
    "synthetic_vision_language",
    "synthetic_vlm",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=(
        "multimodal",
        "vision_language",
        "deterministic",
        "image_conditioned_target",
    ),
    requires=("vocab_size",),
)
class SyntheticVisionLanguageDataset(Dataset):
    """Generate tiny samples with one target token determined by image intensity."""

    def __init__(
        self,
        vocab_size: int,
        num_samples: int = 32,
        sequence_length: int = 8,
        image_size: int = 8,
        num_images: int = 1,
        in_channels: int = 3,
        num_classes: int = 8,
        noise_std: float = 0.05,
        seed: int = 0,
    ):
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        for name, value in (
            ("num_samples", num_samples),
            ("sequence_length", sequence_length),
            ("image_size", image_size),
            ("num_images", num_images),
            ("in_channels", in_channels),
            ("num_classes", num_classes),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if num_classes > vocab_size:
            raise ValueError("num_classes must not exceed vocab_size")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")

        self.vocab_size = int(vocab_size)
        self.num_samples = int(num_samples)
        self.sequence_length = int(sequence_length)
        self.image_size = int(image_size)
        self.num_images = int(num_images)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        class_id = index % self.num_classes
        denominator = max(self.num_classes - 1, 1)
        image_level = 2.0 * class_id / denominator - 1.0
        pixel_values = torch.full(
            (
                self.num_images,
                self.in_channels,
                self.image_size,
                self.image_size,
            ),
            image_level,
            dtype=torch.float32,
        )
        if self.noise_std:
            noise = torch.randn(pixel_values.shape, generator=generator)
            pixel_values = pixel_values + noise * self.noise_std

        input_ids = torch.randint(
            self.vocab_size,
            (self.sequence_length,),
            generator=generator,
        )
        labels = torch.roll(input_ids, shifts=-1)
        labels[-1] = class_id
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.sequence_length, dtype=torch.bool),
            "pixel_values": pixel_values,
            "image_mask": torch.ones(self.num_images, dtype=torch.bool),
            "labels": labels,
            "_sample_id": f"synthetic-vlm-{index}",
            "_image_class": class_id,
        }
