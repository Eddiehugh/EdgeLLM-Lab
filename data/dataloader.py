"""Dataloader helpers and registry."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from core.registry import Registry, build_from_config


DATALOADER_REGISTRY = Registry("dataloader")


@DATALOADER_REGISTRY.register(
    "torch",
    capabilities=("batching", "shuffle", "multiprocessing"),
)
def build_torch_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        **kwargs,
    )


def build_dataloader(
    dataloader_type: str | dict = "torch",
    *,
    dataset: Dataset,
    **kwargs,
) -> DataLoader:
    """Build a dataloader by name."""
    return build_from_config(
        DATALOADER_REGISTRY,
        dataloader_type,
        default_type="torch",
        dataset=dataset,
        **kwargs,
    )
