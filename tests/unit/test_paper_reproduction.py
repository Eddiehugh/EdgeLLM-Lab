from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from core.extensions import load_extensions_from_config
from execution import RunManager
from execution.metadata import JsonMetadataStore
from experiments import STAGE_REGISTRY
from reproduction import (
    ClaimSpec,
    ExpectationSpec,
    PaperStudyManager,
    PaperWorkspace,
    evaluate_claims,
)
from reproduction.specs import PaperConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PaperReproductionTest(unittest.TestCase):
    def test_claim_evaluator_supports_absolute_and_baseline_comparison(self) -> None:
        claim = ClaimSpec(
            id="quality_efficiency",
            statement="The proposed recipe lowers loss and memory.",
            expectations=(
                ExpectationSpec(
                    type="absolute",
                    recipe="proposed",
                    metric="final_train_loss",
                    operator="<=",
                    value=2.0,
                ),
                ExpectationSpec(
                    type="comparison",
                    recipe="proposed",
                    baseline="baseline",
                    metric="peak_memory_mb",
                    mode="ratio",
                    operator="<=",
                    value=0.8,
                ),
            ),
        )
        results = evaluate_claims(
            [claim],
            {
                "baseline": {"final_train_loss": 2.4, "peak_memory_mb": 100},
                "proposed": {"final_train_loss": 1.9, "peak_memory_mb": 75},
            },
        )
        self.assertTrue(results[0].passed)
        self.assertAlmostEqual(results[0].expectations[1].observed, 0.75)

    def test_scaffold_isolated_workspace_and_validate_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = PaperWorkspace(
                root=Path(temporary) / "papers",
                project_root=PROJECT_ROOT,
            )
            directory = workspace.scaffold(
                "test-paper",
                "A Test Paper",
                url="https://example.com/paper",
                authors=["Ada"],
                year=2026,
            )
            self.assertTrue((directory / "paper.yaml").exists())
            self.assertTrue((directory / "implementation" / "components.py").exists())
            self.assertTrue((directory / "tests" / "README.md").exists())
            validation = workspace.validate("test-paper")
            self.assertEqual(validation["suites"], ["smoke"])
            resolved = workspace.resolve_recipe("test-paper", "baseline_smoke")
            self.assertEqual(
                resolved["paper_reproduction"]["paper_id"], "test-paper"
            )
            with self.assertRaises(PaperConfigError):
                workspace.paper_directory("../escape")

    def test_extension_file_can_register_paper_specific_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            name = f"paper_stage_{uuid.uuid4().hex}"
            path = Path(temporary) / "component.py"
            path.write_text(
                "from experiments import ExperimentStage, STAGE_REGISTRY\n"
                f"@STAGE_REGISTRY.register({name!r})\n"
                "class PaperStage(ExperimentStage):\n"
                "    def run(self, context):\n"
                "        context.metrics['paper_stage_ran'] = True\n",
                encoding="utf-8",
            )
            imported = load_extensions_from_config(
                {"extensions": {"paths": [str(path)]}}
            )
            self.assertEqual(imported, [str(path.resolve())])
            self.assertIn(name, STAGE_REGISTRY)

    def test_scaffold_smoke_study_runs_and_writes_claim_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = PaperWorkspace(
                root=root / "papers",
                project_root=PROJECT_ROOT,
            )
            workspace.scaffold("study-paper", "Study Paper")
            run_manager = RunManager(
                metadata_store=JsonMetadataStore(root / "jobs"),
                project_root=PROJECT_ROOT,
            )
            manager = PaperStudyManager(
                workspace=workspace,
                run_manager=run_manager,
                state_root=root / "studies",
            )
            study = manager.run_study(
                "study-paper",
                overrides=[
                    "training.max_steps=1",
                    f"execution.artifact_store.root={root / 'artifacts'}",
                ],
                poll_interval=0.05,
                timeout=30,
            )
            self.assertEqual(study["status"], "completed")
            self.assertEqual(len(study["jobs"]), 2)
            self.assertTrue(study["claims"][0]["passed"])
            self.assertTrue(Path(study["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
