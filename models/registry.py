"""Model registry and factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from core.registry import Registry, build_from_config


MODEL_REGISTRY = Registry[type[nn.Module]]("model")
_BUILTINS_LOADED = False


def _load_builtin_models() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import models.tiny_gpt  # noqa: F401


def build_model(model_type: str | dict = "tiny_gpt", **kwargs: Any) -> nn.Module:
    """Build a model by config name."""
    _load_builtin_models()
    return build_from_config(
        MODEL_REGISTRY,
        model_type,
        default_type="tiny_gpt",
        **kwargs,
    )
