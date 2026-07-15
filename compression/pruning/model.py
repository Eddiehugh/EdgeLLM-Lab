"""Model-level pruning transforms."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from compression.pruning.base import BasePruner
from compression.quantization.base import tensor_storage_bytes
from compression.report import CompressionReport, ModuleCompressionRecord
from compression.selection import ModuleSelector, validate_weight_sharing
from core import model_size_bytes


class _WeightMask(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * self.mask


def prune_linear_modules(
    model: nn.Module,
    pruner: BasePruner,
    *,
    selector: ModuleSelector | None = None,
    inplace: bool = False,
    enforce_mask: bool = False,
    allow_shared_weights: bool = False,
) -> tuple[nn.Module, CompressionReport]:
    """Apply masks to selected Linear weights and report actual sparsity."""

    model_original_bytes = model_size_bytes(model)
    transformed = model if inplace else copy.deepcopy(model)
    selector = selector or ModuleSelector()
    selected = selector.select(transformed, nn.Linear)
    shared_weights = validate_weight_sharing(
        transformed,
        selected,
        allow_shared_weights=allow_shared_weights,
    )
    weights = {name: module.weight.detach() for name, module in selected}
    masks = pruner.compute_masks(weights)
    records = []
    for name, module in selected:
        mask = masks[name].to(device=module.weight.device, dtype=torch.bool)
        original_bytes = tensor_storage_bytes(module.weight)
        pruned = int((mask == 0).sum().item())
        if enforce_mask:
            parametrize.register_parametrization(module, "weight", _WeightMask(mask))
            compressed_bytes = original_bytes + tensor_storage_bytes(mask)
        else:
            with torch.no_grad():
                module.weight.mul_(mask)
            compressed_bytes = original_bytes
        records.append(
            ModuleCompressionRecord(
                name=name,
                module_type=type(module).__name__,
                original_bytes=original_bytes,
                compressed_bytes=compressed_bytes,
                parameter_count=mask.numel(),
                affected_parameters=pruned,
                metadata={
                    "pruner": type(pruner).__name__,
                    "pattern": pruner.pattern,
                    "sparsity": pruned / max(mask.numel(), 1),
                    "mask_enforced": enforce_mask,
                    "shared_weight_owners": list(shared_weights.get(name, ())),
                },
            )
        )
    return transformed, CompressionReport(
        method=type(pruner).__name__,
        records=tuple(records),
        metadata={
            "dense_storage_reduced": False,
            "requires_sparse_backend_for_speedup": True,
            "mask_enforced": enforce_mask,
        },
        model_original_bytes=model_original_bytes,
        model_compressed_bytes=model_size_bytes(transformed),
    )


def remove_pruning_masks(model: nn.Module) -> nn.Module:
    """Bake active parametrized masks into weights for export."""

    for module in model.modules():
        if isinstance(module, nn.Linear) and parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model
