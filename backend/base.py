"""Unified backend interface and registry."""

from __future__ import annotations

from typing import Any

from core.registry import Registry, build_from_config


BACKEND_REGISTRY = Registry[type["InferenceBackend"]]("backend")
_BUILTINS_LOADED = False


class InferenceBackend:
    """Base class for all inference backends."""

    def load_model(self, model_path: str):
        raise NotImplementedError

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError


def _load_builtin_backends() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import backend.executorch_backend  # noqa: F401
    import backend.litert_backend  # noqa: F401
    import backend.llama_cpp_backend  # noqa: F401
    import backend.mlc_backend  # noqa: F401
    import backend.onnx_backend  # noqa: F401
    import backend.torch_backend  # noqa: F401


def build_backend(backend_type: str | dict = "torch", **kwargs: Any) -> InferenceBackend:
    """Build an inference backend by name."""
    _load_builtin_backends()
    return build_from_config(
        BACKEND_REGISTRY,
        backend_type,
        default_type="torch",
        **kwargs,
    )
