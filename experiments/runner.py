"""End-to-end experiment runner built on replaceable pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core import load_extensions_from_config
from core.provenance import capture_environment
from experiments.config import normalize_experiment_config
from experiments.context import ExperimentContext
from experiments.pipeline import ExperimentPipeline
from experiments.report import build_report
from experiments.run_store import RunStore


class ExperimentRunner:
    """Normalize config, execute stages, and persist reproducible artifacts."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = normalize_experiment_config(config)
        load_extensions_from_config(self.config)

    def run(self) -> dict[str, Any]:
        experiment_cfg = dict(self.config.get("experiment", {}))
        name = experiment_cfg.get("name", self.config.get("name", "experiment"))
        output_root = experiment_cfg.get(
            "output_dir", self.config.get("output_dir", "runs")
        )

        run = RunStore(output_root).create_run(str(name))
        config_path = run.write_config(self.config)
        context = ExperimentContext(
            config=self.config,
            run=run,
            environment=capture_environment(),
        )
        context.record_artifact("config", config_path)
        run.write_manifest(context.manifest())

        try:
            pipeline = ExperimentPipeline.from_config(self.config)
            pipeline.run(context)
            context.metrics.update(
                {
                    "run_id": run.run_id,
                    "status": "completed",
                    "stage_runtime_seconds": [
                        {
                            "name": stage.name,
                            "duration_seconds": stage.duration_seconds,
                        }
                        for stage in context.stages
                    ],
                }
            )

            report_path = run.path / "report.md"
            metrics_path = run.path / "metrics.json"
            context.metrics["report_path"] = str(report_path)
            context.record_artifact("metrics", metrics_path)
            context.record_artifact("report", report_path)

            report = build_report(
                title=str(name),
                config=self.config,
                metrics=context.metrics,
            )
            run.write_metrics(context.metrics)
            run.write_text("report.md", report)
            context.finish("completed")
            run.write_manifest(context.manifest())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            context.metrics.update(
                {"run_id": run.run_id, "status": "failed", "error": error}
            )
            metrics_path = run.write_metrics(context.metrics)
            context.record_artifact("metrics", metrics_path)
            context.finish("failed", error=error)
            run.write_manifest(context.manifest())
            raise

        return {"run_dir": str(run.path), "metrics": context.metrics}


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one experiment from a config mapping."""

    return ExperimentRunner(config).run()
