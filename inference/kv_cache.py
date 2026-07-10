"""KV cache abstractions."""

from __future__ import annotations

import torch

from core.registry import Registry, build_from_config


KV_CACHE_REGISTRY = Registry("kv_cache")


@KV_CACHE_REGISTRY.register(
    "append",
    capabilities=("contiguous", "append_only", "autoregressive"),
)
class KVCache:
    """Simple append-only KV cache for autoregressive decoding."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def clear(self) -> None:
        self.k = None
        self.v = None

    def memory_bytes(self) -> int:
        if self.k is None or self.v is None:
            return 0
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


def build_kv_cache(kv_cache_type: str | dict = "append", **kwargs):
    """Build a KV cache by name."""
    return build_from_config(
        KV_CACHE_REGISTRY,
        kv_cache_type,
        default_type="append",
        **kwargs,
    )
