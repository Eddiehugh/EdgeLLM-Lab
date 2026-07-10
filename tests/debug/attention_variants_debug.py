"""Manual debug probe for attention variants.

Run with:

    python3 -m tests.debug.attention_variants_debug
"""

from __future__ import annotations

import torch

from modules.attention import ATTENTION_REGISTRY, build_attention


def main() -> None:
    x = torch.randn(1, 8, 32)
    for name in ATTENTION_REGISTRY.names():
        attention = build_attention(name, hidden_size=32, num_heads=4)
        out = attention(x)
        print(f"{name}: input={tuple(x.shape)} output={tuple(out.shape)}")


if __name__ == "__main__":
    main()
