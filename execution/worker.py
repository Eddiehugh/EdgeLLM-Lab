"""Provider-neutral worker that executes one resolved EdgeLLM experiment."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# ClearML can execute this file as a script before the package is installed.
if __package__ in {None, ""}:  # pragma: no cover - provider bootstrap path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.artifacts import build_artifact_store
from execution.specs import JobSpec, JobState, utc_now
from experiments import run_experiment


def _load_spec(args: argparse.Namespace) -> JobSpec:
    if args.job_spec:
        data = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
    else:
        encoded = args.job_spec_b64 or os.environ.get("EDGELLM_JOB_SPEC_B64")
        if not encoded:
            raise ValueError(
                "Provide --job-spec, --job-spec-b64, or EDGELLM_JOB_SPEC_B64"
            )
        data = json.loads(base64.b64decode(encoded).decode("utf-8"))
    return JobSpec.from_dict(data)


def _write_result(workspace: Path, result: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "worker-result.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_worker(spec: JobSpec) -> dict[str, Any]:
    workspace = Path(spec.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for name, value in spec.env.items():
        os.environ.setdefault(name, value)
    config = dict(spec.experiment_config)
    experiment = dict(config.get("experiment", {}))
    experiment["output_dir"] = str(workspace / "runs")
    config["experiment"] = experiment

    started_at = utc_now()
    try:
        experiment_result = run_experiment(config)
        run_dir = experiment_result["run_dir"]
        artifact_uri = build_artifact_store(spec.artifact_store).publish(
            run_dir, spec.job_id
        )
        result = {
            "job_id": spec.job_id,
            "state": JobState.COMPLETED.value,
            "started_at": started_at,
            "finished_at": utc_now(),
            "run_dir": run_dir,
            "artifact_uri": artifact_uri,
            "metrics": experiment_result["metrics"],
        }
        _write_result(workspace, result)
        return result
    except Exception as exc:
        result = {
            "job_id": spec.job_id,
            "state": JobState.FAILED.value,
            "started_at": started_at,
            "finished_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_result(workspace, result)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgellm-worker")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--job-spec")
    source.add_argument("--job-spec-b64")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_worker(_load_spec(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
