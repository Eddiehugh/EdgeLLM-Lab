"""Prepare-only Colab executor that emits a reproducible notebook."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path

from execution.executors.base import Executor
from execution.specs import JobRecord, JobSpec, JobState


class ColabExecutor(Executor):
    @staticmethod
    def _require_source(spec: JobSpec) -> tuple[str, str]:
        if not spec.source.repo_url or not spec.source.revision:
            raise ValueError(
                "Colab preparation requires source.repo_url and source.revision"
            )
        return spec.source.repo_url, spec.source.revision

    def submit(self, spec: JobSpec) -> JobRecord:
        repo_url, revision = self._require_source(spec)
        workspace = Path(spec.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        data = spec.to_dict()
        data["workspace"] = f"/content/edgellm-jobs/{spec.job_id}"
        data["source"]["project_root"] = "/content/EdgeLLM-Lab"
        remote_spec = JobSpec.from_dict(data)
        encoded = base64.b64encode(
            json.dumps(remote_spec.to_dict()).encode("utf-8")
        ).decode("ascii")
        extras = ",".join(remote_spec.runtime.options.get("extras", []))
        install_target = "/content/EdgeLLM-Lab" + (f"[{extras}]" if extras else "")
        secret_names = list(dict(self.config.get("secrets", {})))
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "colab": {"name": f"EdgeLLM-Lab-{spec.job_id}.ipynb"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
            },
            "cells": [
                self._markdown_cell(
                    "# EdgeLLM-Lab remote job\n"
                    f"Job ID: `{spec.job_id}`. Select a Colab accelerator, "
                    "then run all cells."
                ),
                self._code_cell(
                    "from getpass import getpass\n"
                    "import os\n"
                    f"for name in {secret_names!r}:\n"
                    "    if not os.environ.get(name):\n"
                    "        os.environ[name] = getpass(f'{name}: ')"
                ),
                self._code_cell(
                    f"!git clone {shlex.quote(repo_url)} /content/EdgeLLM-Lab\n"
                    "!git -C /content/EdgeLLM-Lab checkout --detach "
                    f"{shlex.quote(revision)}\n"
                    f"%pip install -q -e {shlex.quote(install_target)}"
                ),
                self._code_cell(
                    "import os\n"
                    f"os.environ['EDGELLM_JOB_SPEC_B64'] = {encoded!r}\n"
                    "%cd /content/EdgeLLM-Lab\n"
                    "!python3 -m execution.worker"
                ),
            ],
        }
        notebook_path = workspace / f"{spec.job_id}.ipynb"
        notebook_path.write_text(
            json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return JobRecord(
            job_id=spec.job_id,
            name=spec.name,
            executor_type="colab",
            state=JobState.PREPARED,
            spec=spec.to_dict(),
            provider_job_id=str(notebook_path),
            message="Notebook prepared; upload it to Colab and run all cells",
            details={"notebook_path": str(notebook_path)},
        )

    @staticmethod
    def _code_cell(source: str) -> dict[str, object]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    @staticmethod
    def _markdown_cell(source: str) -> dict[str, object]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    def status(self, record: JobRecord) -> JobRecord:
        return record

    def logs(self, record: JobRecord, tail: int = 200) -> str:
        del tail
        return record.message or "Colab notebook prepared"

    def cancel(self, record: JobRecord) -> JobRecord:
        return record.update(
            state=JobState.CANCELLED,
            message="Prepared Colab notebook marked as cancelled locally",
        )
