"""Dataset definitions and registry."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from core.registry import Registry, build_from_config


DATASET_REGISTRY = Registry("dataset")


@DATASET_REGISTRY.register("causal_lm")
class CausalLMDataset(Dataset):
    """Fixed-length next-token prediction samples."""

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int):
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(self.token_ids.numel() - self.block_size, 0)

    def __getitem__(self, index: int):
        x = self.token_ids[index : index + self.block_size]
        y = self.token_ids[index + 1 : index + self.block_size + 1]
        return {"input_ids": x, "labels": y}


def build_dataset(dataset_type: str | dict = "causal_lm", **kwargs) -> Dataset:
    """Build a dataset by name."""
    return build_from_config(
        DATASET_REGISTRY,
        dataset_type,
        default_type="causal_lm",
        **kwargs,
    )
