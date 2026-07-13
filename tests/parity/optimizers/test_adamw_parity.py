"""Numerical parity tests for the readable AdamW implementation."""

from __future__ import annotations

import unittest

import torch

from training import build_optimizer


class AdamWParityTest(unittest.TestCase):
    def test_reference_matches_unfused_torch_adamw_updates(self):
        initial = torch.tensor(
            [[0.25, -0.5], [1.0, -1.5]],
            dtype=torch.float64,
        )
        reference_parameter = torch.nn.Parameter(initial.clone())
        torch_parameter = torch.nn.Parameter(initial.clone())
        common = {
            "lr": 2e-3,
            "betas": (0.8, 0.95),
            "eps": 1e-9,
            "weight_decay": 0.1,
        }
        reference_optimizer = build_optimizer(
            {"type": "reference_adamw", **common},
            params=[reference_parameter],
        )
        torch_optimizer = build_optimizer(
            {
                "type": "torch_adamw",
                "foreach": False,
                "fused": False,
                **common,
            },
            params=[torch_parameter],
        )

        for step in range(1, 6):
            gradient = torch.tensor(
                [[0.1 * step, -0.2], [0.05, 0.3 * step]],
                dtype=torch.float64,
            )
            reference_parameter.grad = gradient.clone()
            torch_parameter.grad = gradient.clone()
            reference_optimizer.step()
            torch_optimizer.step()

            torch.testing.assert_close(
                reference_parameter,
                torch_parameter,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_reference_state_dict_round_trip(self):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer = build_optimizer("reference_adamw", params=[parameter], lr=1e-3)
        parameter.grad = torch.tensor([0.5, -0.25], dtype=torch.float64)
        optimizer.step()

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = build_optimizer(
            "reference_adamw",
            params=[restored_parameter],
            lr=1e-3,
        )
        restored.load_state_dict(optimizer.state_dict())

        self.assertEqual(restored.state[restored_parameter]["step"], 1)
        torch.testing.assert_close(
            restored.state[restored_parameter]["exp_avg"],
            optimizer.state[parameter]["exp_avg"],
        )


if __name__ == "__main__":
    unittest.main()
