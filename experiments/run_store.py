"""Run artifact storage."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from core.config import save_config


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "experiment"


class RunStore:
    """Write config, metrics, reports, and checkpoints under one run directory."""

    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, name: str, run_id: str | None = None) -> "Run":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_run_id = run_id or f"{timestamp}-{_slugify(name)}"
        run_id = base_run_id
        path = self.root / run_id
        suffix = 1
        while path.exists():
            suffix += 1
            run_id = f"{base_run_id}-{suffix}"
            path = self.root / run_id
        path.mkdir(parents=True, exist_ok=False)
        latest = self.root / "latest.txt"
        latest.write_text(str(path.resolve()) + "\n", encoding="utf-8")
        return Run(path=path, run_id=run_id)


class Run:
    """A concrete run artifact directory."""

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id

    def write_json(self, relative_path: str, data: Any) -> Path:
        output_path = self.path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return output_path

    def write_text(self, relative_path: str, text: str) -> Path:
        output_path = self.path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def write_config(self, config: dict[str, Any]) -> Path:
        return self.write_config_as("config.yaml", config)

    def write_config_as(self, relative_path: str, config: dict[str, Any]) -> Path:
        output_path = self.path / relative_path
        save_config(config, output_path)
        return output_path

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        return self.write_json("metrics.json", metrics)

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        return self.write_json("manifest.json", manifest)

    def write_checkpoint(self, model: torch.nn.Module, relative_path: str = "checkpoint.pt") -> Path:
        output_path = self.path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        return output_path
