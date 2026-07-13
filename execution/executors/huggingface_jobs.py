"""Hugging Face Jobs executor using huggingface_hub's official API."""

from __future__ import annotations

import base64
import json
import os
import shlex
from typing import Any

from execution.executors.base import Executor
from execution.specs import JobRecord, JobSpec, JobState


class HuggingFaceJobsExecutor(Executor):
    @staticmethod
    def _hub():
        try:
            import huggingface_hub
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face Jobs require the 'hf' extra: pip install -e '.[hf]'"
            ) from exc
        return huggingface_hub

    @staticmethod
    def _require_source(spec: JobSpec) -> tuple[str, str]:
        if not spec.source.repo_url or not spec.source.revision:
            raise ValueError(
                "Hugging Face Jobs require source.repo_url and source.revision"
            )
        return spec.source.repo_url, spec.source.revision

    def _token(self) -> str | None:
        return os.environ.get(str(self.config.get("token_env", "HF_TOKEN")))

    def _secrets(self) -> dict[str, str]:
        resolved = {}
        for remote_name, local_name in dict(self.config.get("secrets", {})).items():
            value = os.environ.get(str(local_name))
            if value is None:
                raise ValueError(
                    f"Environment variable '{local_name}' is required for secret "
                    f"'{remote_name}'"
                )
            resolved[str(remote_name)] = value
        return resolved

    @staticmethod
    def _remote_spec(spec: JobSpec) -> JobSpec:
        data = spec.to_dict()
        data["workspace"] = f"/tmp/edgellm-jobs/{spec.job_id}"
        data["source"]["project_root"] = "/tmp/edgellm-source"
        return JobSpec.from_dict(data)

    def submit(self, spec: JobSpec) -> JobRecord:
        repo_url, revision = self._require_source(spec)
        if not spec.runtime.image:
            raise ValueError(
                "runtime.image is required for Hugging Face Jobs, for example "
                "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime"
            )
        remote_spec = self._remote_spec(spec)
        encoded = base64.b64encode(
            json.dumps(remote_spec.to_dict()).encode("utf-8")
        ).decode("ascii")
        source_dir = remote_spec.source.project_root
        extras = ",".join(remote_spec.runtime.options.get("extras", []))
        install_target = source_dir + (f"[{extras}]" if extras else "")
        script = " && ".join(
            [
                f"git clone {shlex.quote(repo_url)} {shlex.quote(source_dir)}",
                f"git -C {shlex.quote(source_dir)} checkout --detach "
                f"{shlex.quote(revision)}",
                f"{shlex.quote(spec.runtime.python)} -m pip install -e "
                f"{shlex.quote(install_target)}",
                f"cd {shlex.quote(source_dir)}",
                f"{shlex.quote(spec.runtime.python)} -m execution.worker",
            ]
        )
        environment = dict(spec.env)
        environment["EDGELLM_JOB_SPEC_B64"] = encoded
        kwargs: dict[str, Any] = {
            "image": spec.runtime.image,
            "command": ["bash", "-lc", script],
            "flavor": self.config.get("flavor", "a10g-small"),
            "env": environment,
            "secrets": self._secrets(),
        }
        for key in ("namespace", "timeout"):
            if self.config.get(key) is not None:
                kwargs[key] = self.config[key]
        if self._token():
            kwargs["token"] = self._token()
        job = self._hub().run_job(**kwargs)
        provider_id = str(getattr(job, "id", getattr(job, "job_id", "")))
        if not provider_id:
            raise RuntimeError("Hugging Face Jobs did not return a job id")
        return JobRecord(
            job_id=spec.job_id,
            name=spec.name,
            executor_type="huggingface_jobs",
            state=JobState.QUEUED,
            spec=spec.to_dict(),
            provider_job_id=provider_id,
            url=getattr(job, "url", None),
        )

    @staticmethod
    def _stage(info: Any) -> str:
        status = getattr(info, "status", None)
        stage = getattr(status, "stage", status)
        return str(getattr(stage, "value", stage)).lower().split(".")[-1]

    @staticmethod
    def _map_state(stage: str) -> JobState:
        if stage in {"pending", "queued", "starting", "scheduling"}:
            return JobState.QUEUED
        if stage in {"running"}:
            return JobState.RUNNING
        if stage in {"completed", "succeeded", "success"}:
            return JobState.COMPLETED
        if stage in {"cancelled", "canceled"}:
            return JobState.CANCELLED
        if stage in {"failed", "error", "deleted"}:
            return JobState.FAILED
        return JobState.UNKNOWN

    def status(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        kwargs: dict[str, Any] = {}
        if self.config.get("namespace"):
            kwargs["namespace"] = self.config["namespace"]
        if self._token():
            kwargs["token"] = self._token()
        info = self._hub().inspect_job(record.provider_job_id, **kwargs)
        status = getattr(info, "status", None)
        message = getattr(status, "message", None)
        return record.update(state=self._map_state(self._stage(info)), message=message)

    def logs(self, record: JobRecord, tail: int = 200) -> str:
        kwargs: dict[str, Any] = {"follow": False, "tail": tail}
        if self.config.get("namespace"):
            kwargs["namespace"] = self.config["namespace"]
        if self._token():
            kwargs["token"] = self._token()
        output = self._hub().fetch_job_logs(record.provider_job_id, **kwargs)
        if isinstance(output, str):
            return output
        return "\n".join(str(line).rstrip("\n") for line in output)

    def cancel(self, record: JobRecord) -> JobRecord:
        if record.state.terminal:
            return record
        kwargs: dict[str, Any] = {}
        if self.config.get("namespace"):
            kwargs["namespace"] = self.config["namespace"]
        if self._token():
            kwargs["token"] = self._token()
        self._hub().cancel_job(record.provider_job_id, **kwargs)
        return record.update(state=JobState.CANCELLED, message="Cancellation requested")
