"""ClearML executor backed by Task and Agent queues."""

from __future__ import annotations

import base64
import json
from typing import Any

from execution.executors.base import Executor
from execution.specs import JobRecord, JobSpec, JobState


class ClearMLExecutor(Executor):
    @staticmethod
    def _task_class():
        try:
            from clearml import Task
        except ImportError as exc:
            raise RuntimeError(
                "ClearML execution requires the 'clearml' extra: "
                "pip install -e '.[clearml]'"
            ) from exc
        return Task

    @staticmethod
    def _require_source(spec: JobSpec) -> tuple[str, str]:
        if not spec.source.repo_url or not spec.source.revision:
            raise ValueError("ClearML requires source.repo_url and source.revision")
        return spec.source.repo_url, spec.source.revision

    @staticmethod
    def _remote_spec(spec: JobSpec) -> JobSpec:
        data = spec.to_dict()
        data["workspace"] = f"/tmp/edgellm-jobs/{spec.job_id}"
        data["source"]["project_root"] = "."
        return JobSpec.from_dict(data)

    def submit(self, spec: JobSpec) -> JobRecord:
        repo_url, revision = self._require_source(spec)
        remote_spec = self._remote_spec(spec)
        encoded = base64.b64encode(
            json.dumps(remote_spec.to_dict()).encode("utf-8")
        ).decode("ascii")
        Task = self._task_class()
        create_kwargs: dict[str, Any] = {
            "project_name": self.config.get("project_name", "EdgeLLM-Lab"),
            "task_name": spec.name,
            "repo": repo_url,
            "commit": revision,
            "script": "execution/worker.py",
            "working_directory": ".",
            "argparse_args": [("--job-spec-b64", encoded)],
        }
        if spec.runtime.image:
            create_kwargs["docker"] = spec.runtime.image
        if self.config.get("packages"):
            create_kwargs["packages"] = list(self.config["packages"])
        task = Task.create(**create_kwargs)
        Task.enqueue(
            task=task,
            queue_name=self.config.get("queue", "default"),
        )
        return JobRecord(
            job_id=spec.job_id,
            name=spec.name,
            executor_type="clearml",
            state=JobState.QUEUED,
            spec=spec.to_dict(),
            provider_job_id=str(task.id),
            url=self._task_url(task),
        )

    @staticmethod
    def _task_url(task: Any) -> str | None:
        for method_name in ("get_output_log_web_page", "get_task_output_log_web_page"):
            method = getattr(task, method_name, None)
            if callable(method):
                try:
                    return str(method())
                except Exception:
                    return None
        return None

    @staticmethod
    def _state(status: Any) -> JobState:
        value = str(getattr(status, "value", status)).lower().split(".")[-1]
        if value in {"created", "queued", "pending", "published"}:
            return JobState.QUEUED
        if value in {"in_progress", "running"}:
            return JobState.RUNNING
        if value in {"completed", "closed"}:
            return JobState.COMPLETED
        if value in {"stopped", "cancelled", "canceled"}:
            return JobState.CANCELLED
        if value in {"failed"}:
            return JobState.FAILED
        return JobState.UNKNOWN

    def _task(self, record: JobRecord):
        return self._task_class().get_task(task_id=record.provider_job_id)

    def status(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        task = self._task(record)
        return record.update(state=self._state(task.status))

    def logs(self, record: JobRecord, tail: int = 200) -> str:
        task = self._task(record)
        output = task.get_reported_console_output(number_of_reports=1)
        if isinstance(output, list):
            text = "\n".join(str(item) for item in output)
        else:
            text = str(output or "")
        return "\n".join(text.splitlines()[-tail:])

    def cancel(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        self._task(record).abort()
        return record.update(state=JobState.CANCELLED, message="Cancellation requested")
