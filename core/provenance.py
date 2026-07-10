"""Reproducibility metadata captured for every experiment run."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def capture_environment(root: str | Path | None = None) -> dict[str, Any]:
    """Capture stable environment facts without collecting user secrets."""

    project_root = Path(root or Path.cwd()).resolve()
    git_status = _git_output(project_root, "status", "--porcelain")
    cuda_version = getattr(torch.version, "cuda", None)
    mps_backend = getattr(torch.backends, "mps", None)

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": cuda_version,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(mps_backend and mps_backend.is_available()),
        },
        "project": {
            "root": str(project_root),
            "git_revision": _git_output(project_root, "rev-parse", "HEAD"),
            "git_dirty": bool(git_status) if git_status is not None else None,
        },
        "executable": sys.executable,
    }
