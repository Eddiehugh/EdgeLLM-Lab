"""Dataset definitions and registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from core.registry import Registry, build_from_config


DATASET_REGISTRY = Registry("dataset")
_BUILTINS_LOADED = False


@DATASET_REGISTRY.register(
    "causal_lm",
    capabilities=("next_token_prediction", "fixed_length"),
    requires=("token_ids", "block_size"),
)
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


def load_builtin_datasets() -> None:
    """Load datasets that live in independent technique files."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import data.datasets.synthetic_vision_language  # noqa: F401


def build_dataset(
    dataset_type: str | Mapping[str, Any] = "causal_lm",
    **kwargs,
) -> Dataset:
    """Build a dataset by name."""
    load_builtin_datasets()
    if isinstance(dataset_type, Mapping) and "name" in dataset_type:
        dataset_type = dict(dataset_type)
        dataset_type.setdefault("type", dataset_type.pop("name"))
    return build_from_config(
        DATASET_REGISTRY,
        dataset_type,
        default_type="causal_lm",
        **kwargs,
    )
