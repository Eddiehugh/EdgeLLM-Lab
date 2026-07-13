"""Tests for structured optimizer selection and parameter policies."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from training import (
    OPTIMIZER_REGISTRY,
    PARAM_GROUP_POLICY_REGISTRY,
    ReferenceAdamW,
    build_optimizer,
)


class OptimizerApiTest(unittest.TestCase):
    def test_legacy_type_config_remains_supported(self):
        model = nn.Linear(4, 2)

        optimizer = build_optimizer(
            {"type": "adamw", "lr": 1e-3},
            params=model.parameters(),
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_structured_config_selects_algorithm_backend_and_policy(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))

        optimizer = build_optimizer(
            {
                "algorithm": "adamw",
                "implementation": "reference",
                "lr": 1e-3,
                "weight_decay": 0.2,
                "param_group_policy": {"type": "decay_by_dimension"},
            },
            model=model,
        )

        self.assertIsInstance(optimizer, ReferenceAdamW)
        groups = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(groups), {"decay", "no_decay"})
        self.assertEqual(groups["decay"]["weight_decay"], 0.2)
        self.assertEqual(groups["no_decay"]["weight_decay"], 0.0)
        self.assertTrue(all(parameter.ndim >= 2 for parameter in groups["decay"]["params"]))
        self.assertTrue(all(parameter.ndim < 2 for parameter in groups["no_decay"]["params"]))

    def test_named_policy_requires_model_context(self):
        model = nn.Linear(4, 2)

        with self.assertRaisesRegex(ValueError, "require model"):
            build_optimizer(
                {
                    "type": "adamw",
                    "param_group_policy": "decay_by_dimension",
                },
                params=model.parameters(),
            )

    def test_optimizer_axes_are_independently_discoverable(self):
        self.assertIn("reference_adamw", OPTIMIZER_REGISTRY.names())
        self.assertIn("torch_adamw", OPTIMIZER_REGISTRY.names())
        self.assertIn("decay_by_dimension", PARAM_GROUP_POLICY_REGISTRY.names())
        self.assertEqual(
            OPTIMIZER_REGISTRY.describe("reference_adamw")["level"],
            "level_1",
        )


if __name__ == "__main__":
    unittest.main()
