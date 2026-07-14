"""Provider-neutral worker that executes one resolved EdgeLLM experiment."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from glob import glob
from pathlib import Path
from typing import Any

# ClearML can execute this file as a script before the package is installed.
if __package__ in {None, ""}:  # pragma: no cover - provider bootstrap path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.artifacts import build_artifact_store
from execution.specs import CommandSpec, JobSpec, JobState, utc_now
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


def _expanded_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded if expanded.is_absolute() else base / expanded


def _append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message)
        log_file.flush()


def _run_command(
    command: CommandSpec, cwd: Path, log_path: Path
) -> dict[str, Any]:
    marker = command.skip_if_exists
    if marker and _expanded_path(marker, cwd).exists():
        message = f"Skipping setup command; marker exists: {marker}\n"
        print(message, end="", flush=True)
        _append_log(log_path, message)
        return {
            "argv": list(command.argv),
            "status": "skipped",
            "duration_seconds": 0.0,
        }

    argv = [os.path.expandvars(os.path.expanduser(value)) for value in command.argv]
    message = f"Running in {cwd}: {argv!r}\n"
    print(message, end="", flush=True)
    _append_log(log_path, message)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, argv)
    return {
        "argv": argv,
        "status": "completed",
        "duration_seconds": time.monotonic() - started,
    }


def _checkout_external_source(spec: JobSpec, workspace: Path) -> tuple[Path, str]:
    workload = spec.workload
    source = workload.source
    if source is None or not source.repo_url or not source.revision:
        raise ValueError("External workload requires a pinned Git source")
    checkout = workspace / "projects" / (workload.integration or "external")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", source.repo_url, str(checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", source.revision],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return checkout.resolve(), revision


def _collect_external_artifacts(
    patterns: tuple[str, ...], project_dir: Path, run_dir: Path
) -> list[dict[str, str]]:
    artifact_dir = run_dir / "artifacts"
    collected: list[dict[str, str]] = []
    for pattern in patterns:
        expanded = os.path.expandvars(os.path.expanduser(pattern))
        search_pattern = (
            expanded if os.path.isabs(expanded) else str(project_dir / expanded)
        )
        matches = sorted(Path(match).resolve() for match in glob(search_pattern))
        if not matches:
            raise FileNotFoundError(f"Declared artifact path did not match: {pattern}")
        for source in matches:
            destination = artifact_dir / source.name
            if destination.exists():
                raise FileExistsError(
                    f"Artifact basename collision for {source}: {destination.name}"
                )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            collected.append(
                {"source": str(source), "path": str(destination.relative_to(run_dir))}
            )
    return collected


def _run_external_workload(spec: JobSpec, workspace: Path) -> dict[str, Any]:
    workload = spec.workload
    if workload.command is None:
        raise ValueError("External workload command is missing")
    checkout, revision = _checkout_external_source(spec, workspace)
    project_dir = (checkout / workload.working_directory).resolve()
    try:
        project_dir.relative_to(checkout)
    except ValueError as exc:
        raise ValueError(
            "External workload working_directory escapes its checkout"
        ) from exc
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {project_dir}")

    run_dir = workspace / "runs" / (workload.integration or "external")
    run_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_dir / "command.log"
    setup_results = [
        _run_command(command, project_dir, command_log)
        for command in workload.setup
    ]
    command_result = _run_command(workload.command, project_dir, command_log)
    collected = _collect_external_artifacts(
        workload.artifacts, project_dir, run_dir
    )
    metrics = {
        "status": "completed",
        "workload_type": workload.type,
        "integration": workload.integration,
        "revision": revision,
        "setup_commands": len(setup_results),
        "setup_duration_seconds": sum(
            float(result["duration_seconds"]) for result in setup_results
        ),
        "command_duration_seconds": command_result["duration_seconds"],
        "artifact_count": len(collected),
    }
    run_metadata = {
        "job_id": spec.job_id,
        "name": spec.name,
        "source": workload.source.__dict__ if workload.source else None,
        "revision": revision,
        "working_directory": workload.working_directory,
        "command_log": command_log.name,
        "setup": setup_results,
        "command": command_result,
        "artifacts": collected,
        "metrics": metrics,
    }
    (run_dir / "external-run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"job_id": spec.job_id, "revision": revision, "artifacts": collected},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        f"# {spec.name}\n\n"
        f"- Integration: `{workload.integration}`\n"
        f"- Revision: `{revision}`\n"
        f"- Command duration: {command_result['duration_seconds']:.2f} seconds\n"
        f"- Collected artifacts: {len(collected)}\n",
        encoding="utf-8",
    )
    return {"run_dir": str(run_dir), "metrics": metrics}


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
        experiment_result = (
            _run_external_workload(spec, workspace)
            if spec.workload.type == "external_project"
            else run_experiment(config)
        )
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
