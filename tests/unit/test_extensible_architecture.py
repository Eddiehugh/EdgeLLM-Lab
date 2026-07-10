"""Tests for registry metadata and the extensible experiment pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core import Maturity, ProjectLevel, Registry
from experiments import (
    ExperimentConfigError,
    ExperimentStage,
    STAGE_REGISTRY,
    normalize_experiment_config,
    run_experiment,
)


class ExtensibleArchitectureTest(unittest.TestCase):
    def test_registry_exposes_canonical_metadata_and_aliases(self):
        registry = Registry("demo")

        @registry.register(
            "reference",
            "ref",
            level=ProjectLevel.LEARN,
            maturity=Maturity.VERIFIED,
            capabilities=("forward", "backward"),
        )
        class ReferenceComponent:
            """Reference implementation."""

        self.assertEqual(registry.canonical_names(), ("reference",))
        self.assertIs(registry.get("ref"), ReferenceComponent)
        self.assertTrue(registry.get_spec("reference").supports("forward"))
        self.assertEqual(registry.describe("ref")["maturity"], "verified")
        self.assertEqual(registry.describe("ref")["source"], __name__)

    def test_config_normalization_adds_a_versioned_default_pipeline(self):
        config = normalize_experiment_config(
            {
                "model": {"hidden_size": 16},
                "training": {"max_steps": 1},
            }
        )

        self.assertEqual(config["schema_version"], 1)
        self.assertIn("build_model", config["pipeline"]["stages"])
        self.assertIn("train", config["pipeline"]["stages"])

        with self.assertRaises(ExperimentConfigError):
            normalize_experiment_config(
                {
                    "schema_version": 999,
                    "training": {"max_steps": 0},
                }
            )

    def test_custom_stages_extend_runner_without_runner_changes(self):
        @STAGE_REGISTRY.register(
            "test_producer",
            override=True,
            level=ProjectLevel.EXPERIMENT,
            maturity=Maturity.VERIFIED,
        )
        class ProducerStage(ExperimentStage):
            provides = ("test_value",)

            def run(self, context):
                context.provide("test_value", 21)

        @STAGE_REGISTRY.register(
            "test_consumer",
            override=True,
            level=ProjectLevel.EXPERIMENT,
            maturity=Maturity.VERIFIED,
        )
        class ConsumerStage(ExperimentStage):
            requires = ("test_value",)

            def run(self, context):
                context.metrics["custom_result"] = context.require("test_value") * 2

        self.assertEqual(
            STAGE_REGISTRY.describe("test_producer")["provides"],
            ("test_value",),
        )
        self.assertEqual(
            STAGE_REGISTRY.describe("test_consumer")["requires"],
            ("test_value",),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiment(
                {
                    "experiment": {"name": "custom-pipeline", "output_dir": tmpdir},
                    "pipeline": {"stages": ["test_producer", "test_consumer"]},
                }
            )
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text())

            self.assertEqual(result["metrics"]["custom_result"], 42)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                [stage["name"] for stage in manifest["stages"]],
                ["test_producer", "test_consumer"],
            )
            self.assertTrue((run_dir / "config.yaml").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "report.md").exists())

    def test_invalid_pipeline_order_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "Invalid pipeline order"):
                run_experiment(
                    {
                        "experiment": {"name": "invalid-order", "output_dir": tmpdir},
                        "pipeline": {"stages": ["train"]},
                    }
                )

            run_dirs = [path for path in Path(tmpdir).iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("Invalid pipeline order", manifest["error"])


if __name__ == "__main__":
    unittest.main()
