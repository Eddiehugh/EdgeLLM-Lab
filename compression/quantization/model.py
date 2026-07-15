"""Model-level weight-only quantization transforms."""

from __future__ import annotations

import copy

import torch.nn as nn

from compression.quantization.base import (
    TensorQuantizer,
    quantization_error,
    tensor_storage_bytes,
)
from compression.quantization.quant_linear import ReferenceQuantizedLinear
from compression.report import CompressionReport, ModuleCompressionRecord
from compression.selection import (
    ModuleSelector,
    replace_module,
    validate_weight_sharing,
)
from core import model_size_bytes


def quantize_linear_modules(
    model: nn.Module,
    quantizer: TensorQuantizer,
    *,
    selector: ModuleSelector | None = None,
    inplace: bool = False,
    allow_shared_weights: bool = False,
) -> tuple[nn.Module, CompressionReport]:
    """Replace selected Linear modules with portable weight-only references."""

    model_original_bytes = model_size_bytes(model)
    transformed = model if inplace else copy.deepcopy(model)
    selector = selector or ModuleSelector()
    records = []
    selected = selector.select(transformed, nn.Linear)
    shared_weights = validate_weight_sharing(
        transformed,
        selected,
        allow_shared_weights=allow_shared_weights,
    )
    for name, module in selected:
        reference_weight = module.weight.detach()
        replacement = ReferenceQuantizedLinear.from_float(module, quantizer)
        replacement.train(module.training)
        reconstructed = replacement.dequantized_weight()
        original_bytes = tensor_storage_bytes(reference_weight)
        records.append(
            ModuleCompressionRecord(
                name=name,
                module_type=type(module).__name__,
                original_bytes=original_bytes,
                compressed_bytes=replacement.weight_storage_bytes,
                parameter_count=reference_weight.numel(),
                affected_parameters=reference_weight.numel(),
                metadata={
                    "quantizer": type(quantizer).__name__,
                    "bits": replacement.weight_bits,
                    "packed": replacement.packed,
                    "group_size": replacement.group_size,
                    "shared_weight_owners": list(shared_weights.get(name, ())),
                    **quantization_error(reference_weight, reconstructed),
                },
            )
        )
        replace_module(transformed, name, replacement)

    return transformed, CompressionReport(
        method=type(quantizer).__name__,
        records=tuple(records),
        metadata={"reference_forward": True, "latency_accelerated": False},
        model_original_bytes=model_original_bytes,
        model_compressed_bytes=model_size_bytes(transformed),
    )
