"""Attention implementations.

Each attention technique lives in its own file. This package only exposes the
registry and imports built-ins so their decorators run.
"""

from modules.attention.registry import ATTENTION_REGISTRY, build_attention

from modules.attention.gqa import GroupedQueryAttention
from modules.attention.mha import MultiHeadAttention
from modules.attention.mla import MultiHeadLatentAttention
from modules.attention.mqa import MultiQueryAttention
from modules.attention.sliding_window import SlidingWindowAttention
from modules.attention.sparse import TopKSparseAttention

__all__ = [
    "ATTENTION_REGISTRY",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "MultiHeadLatentAttention",
    "MultiQueryAttention",
    "SlidingWindowAttention",
    "TopKSparseAttention",
    "build_attention",
]
