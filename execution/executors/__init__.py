"""Executor implementations and factory."""

from __future__ import annotations

from execution.executors.base import Executor
from execution.executors.clearml import ClearMLExecutor
from execution.executors.colab import ColabExecutor
from execution.executors.huggingface_jobs import HuggingFaceJobsExecutor
from execution.executors.local import LocalExecutor
from execution.executors.ssh import SSHExecutor


def build_executor(executor_type: str, config: dict | None = None) -> Executor:
    executors = {
        "local": LocalExecutor,
        "ssh": SSHExecutor,
        "autodl": SSHExecutor,
        "huggingface_jobs": HuggingFaceJobsExecutor,
        "hf_jobs": HuggingFaceJobsExecutor,
        "clearml": ClearMLExecutor,
        "colab": ColabExecutor,
    }
    try:
        executor_class = executors[executor_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown executor '{executor_type}'. Available: {', '.join(executors)}"
        ) from exc
    return executor_class(config)


__all__ = [
    "ClearMLExecutor",
    "ColabExecutor",
    "Executor",
    "HuggingFaceJobsExecutor",
    "LocalExecutor",
    "SSHExecutor",
    "build_executor",
]
