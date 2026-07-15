"""Model I/O helpers shared by text and future multimodal training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensor-bearing model inputs without changing structure."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def prepare_model_inputs(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    input_keys: Sequence[str] | None = None,
    label_key: str = "labels",
) -> tuple[dict[str, Any], Any]:
    """Split labels from keyword model inputs and move both to the device."""

    labels = move_to_device(batch.get(label_key), device)
    keys = list(input_keys) if input_keys is not None else [
        key for key in batch if key != label_key and not key.startswith("_")
    ]
    missing = [key for key in keys if key not in batch]
    if missing:
        raise KeyError("Configured model input keys are missing: " + ", ".join(missing))
    inputs = {key: move_to_device(batch[key], device) for key in keys}
    if not inputs:
        raise ValueError("Batch does not contain any model inputs")
    return inputs, labels


def extract_logits(output: Any) -> torch.Tensor:
    """Extract logits from Tensor, mapping, or object-style model outputs."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError("Model output must be a Tensor or expose Tensor logits")
