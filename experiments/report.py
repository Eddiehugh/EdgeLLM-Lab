"""Markdown report generation for experiment runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(
    *,
    title: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    """Build a compact Markdown report for one experiment run."""

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"| `{key}` | {_format_metric_value(value)} |")

    lines.extend(
        [
            "",
            "## Config",
            "",
            "```json",
            json.dumps(config, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
