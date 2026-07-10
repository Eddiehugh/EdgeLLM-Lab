"""Unit tests for independently packaged attention variants."""

from __future__ import annotations

import unittest

import torch

from modules.attention import ATTENTION_REGISTRY, build_attention


class AttentionVariantsTest(unittest.TestCase):
    def test_registered_attention_variants_keep_shape(self):
        x = torch.randn(2, 8, 32)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
        mask = mask.view(1, 1, 8, 8)

        variants = {
            "mha": {},
            "mqa": {},
            "gqa": {"num_kv_heads": 2},
            "mla": {"latent_size": 16},
            "sliding_window": {"window_size": 4},
            "topk_sparse": {"top_k": 4},
        }

        for name, kwargs in variants.items():
            with self.subTest(name=name):
                attention = build_attention(
                    {"type": name, **kwargs},
                    hidden_size=32,
                    num_heads=4,
                )
                out = attention(x, mask=mask)
                self.assertEqual(out.shape, x.shape)

        self.assertTrue(set(variants).issubset(set(ATTENTION_REGISTRY.names())))


if __name__ == "__main__":
    unittest.main()
