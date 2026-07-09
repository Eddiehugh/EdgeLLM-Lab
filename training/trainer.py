"""Compatibility training entry point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments import run_experiment


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run a training experiment from a config mapping."""

    return run_experiment(config)
