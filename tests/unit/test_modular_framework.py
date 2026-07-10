"""Framework-level smoke tests."""

from __future__ import annotations

import tempfile
import unittest

import torch

from data import build_dataloader, build_dataset, build_tokenizer
from experiments import run_experiment
from integrations import build_integration, integration_snapshot
from models import build_model
from training import build_loss


class ModularFrameworkTest(unittest.TestCase):
    def test_component_factories_build_a_trainable_batch(self):
        tokenizer = build_tokenizer("char")
        token_ids = tokenizer.encode("hello modular framework\n" * 4)
        dataset = build_dataset("causal_lm", token_ids=token_ids, block_size=8)
        dataloader = build_dataloader("torch", dataset=dataset, batch_size=2)
        batch = next(iter(dataloader))

        model = build_model(
            "tiny_gpt",
            vocab_size=256,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            max_position_embeddings=8,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        logits = model(batch["input_ids"])
        loss = build_loss("causal_lm")(logits, labels=batch["labels"])

        self.assertEqual(logits.shape, (2, 8, 256))
        self.assertTrue(torch.isfinite(loss))

    def test_experiment_runner_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiment(
                {
                    "experiment": {"name": "unit-smoke", "output_dir": tmpdir},
                    "runtime": {"device": "cpu", "seed": 7},
                    "model": {
                        "name": "tiny_gpt",
                        "vocab_size": 256,
                        "hidden_size": 32,
                        "num_layers": 1,
                        "num_heads": 4,
                        "max_position_embeddings": 8,
                    },
                    "data": {
                        "tokenizer_type": "char",
                        "text": "unit smoke test text\n" * 8,
                        "block_size": 8,
                    },
                    "loss": {"type": "causal_lm"},
                    "training": {
                        "batch_size": 2,
                        "max_steps": 1,
                        "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.0},
                        "scheduler": {"type": "constant"},
                    },
                }
            )

        self.assertEqual(result["metrics"]["steps"], 1)
        self.assertIn("final_train_loss", result["metrics"])

    def test_builtin_integrations_are_discoverable(self):
        snapshot = integration_snapshot()

        self.assertIn("nanogpt", snapshot)
        self.assertIn("tinyllama", snapshot)
        self.assertIn("llama_cpp", snapshot)
        self.assertEqual(build_integration("nanogpt").name, "nanogpt")
        self.assertTrue(
            snapshot["nanogpt"]["expected_repo_path"].endswith(
                "external_projects/nanogpt/repo"
            )
        )


if __name__ == "__main__":
    unittest.main()
