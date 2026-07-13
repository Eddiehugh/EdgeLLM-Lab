"""Asynchronous local process executor."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

from execution.executors.base import Executor, read_tail
from execution.runtimes import build_runtime
from execution.specs import JobRecord, JobSpec, JobState


_LOCAL_PROCESSES: dict[int, subprocess.Popen] = {}


class LocalExecutor(Executor):
    def submit(self, spec: JobSpec) -> JobRecord:
        workspace = Path(spec.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        spec_path = workspace / "job-spec.json"
        spec_path.write_text(
            json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log_path = workspace / "worker.log"
        runtime = build_runtime(spec.runtime)
        command = runtime.worker_command(
            spec.source.project_root, workspace, spec_path
        )
        environment = os.environ.copy()
        environment.update(spec.env)
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=spec.source.project_root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _LOCAL_PROCESSES[process.pid] = process
        return JobRecord(
            job_id=spec.job_id,
            name=spec.name,
            executor_type="local",
            state=JobState.RUNNING,
            spec=spec.to_dict(),
            provider_job_id=str(process.pid),
            log_path=str(log_path),
            details={"pid": process.pid, "command": command},
        )

    def status(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        spec = JobSpec.from_dict(record.spec)
        result_path = Path(spec.workspace) / "worker-result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            process = _LOCAL_PROCESSES.pop(int(record.details["pid"]), None)
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            return record.update(
                state=JobState(result["state"]),
                artifact_uri=result.get("artifact_uri", record.artifact_uri),
                message=result.get("error"),
                details={**record.details, "worker_result": result},
            )

        pid = int(record.details["pid"])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return record.update(
                state=JobState.FAILED,
                message="Worker exited without writing worker-result.json",
            )
        except PermissionError:
            pass
        return record.update(state=JobState.RUNNING)

    def logs(self, record: JobRecord, tail: int = 200) -> str:
        return read_tail(record.log_path or "", tail)

    def cancel(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        pid = int(record.details["pid"])
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process = _LOCAL_PROCESSES.pop(pid, None)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        return record.update(state=JobState.CANCELLED, message="Cancelled locally")
