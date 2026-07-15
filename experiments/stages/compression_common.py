"""Configuration helpers shared by compression pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from compression import CompressionReport


def compression_passes(configured: Any, field_name: str) -> list[str | dict[str, Any]]:
    """Normalize one component config or an ordered list of pass configs."""

    if isinstance(configured, str):
        return [configured]
    if isinstance(configured, Mapping):
        return [dict(configured)]
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        passes = []
        for index, item in enumerate(configured):
            if isinstance(item, str):
                passes.append(item)
            elif isinstance(item, Mapping):
                passes.append(dict(item))
            else:
                raise TypeError(
                    f"{field_name}[{index}] must be a component name or mapping"
                )
        if not passes:
            raise ValueError(f"{field_name} must contain at least one pass")
        return passes
    raise TypeError(f"{field_name} must be a component name, mapping, or list")


def split_compression_pass(
    configured: str | Mapping[str, Any],
    *,
    transform_fields: tuple[str, ...],
) -> tuple[str | None, str | dict[str, Any], dict[str, Any]]:
    """Separate registry constructor fields from model transform controls."""

    if isinstance(configured, str):
        return None, configured, {}
    component_cfg = dict(configured)
    pass_id = component_cfg.pop("id", None)
    if pass_id is not None and (not isinstance(pass_id, str) or not pass_id):
        raise ValueError("Compression pass id must be a non-empty string")
    transform_cfg = {
        field: component_cfg.pop(field)
        for field in transform_fields
        if field in component_cfg
    }
    return pass_id, component_cfg, transform_cfg


def combined_report(
    kind: str,
    reports: Sequence[CompressionReport],
) -> CompressionReport:
    return CompressionReport.combine(f"{kind}_plan", reports)


def unique_pass_label(
    pass_id: str | None,
    index: int,
    seen_labels: set[str],
) -> str:
    label = pass_id or str(index)
    if label in seen_labels:
        raise ValueError(f"Duplicate compression pass id '{label}'")
    seen_labels.add(label)
    return label


def reject_overlapping_modules(
    seen: set[str],
    report: CompressionReport,
    *,
    pass_label: str,
) -> None:
    selected = {record.name for record in report.records}
    overlap = sorted(seen & selected)
    if overlap:
        raise ValueError(
            f"Compression pass '{pass_label}' overlaps earlier modules: "
            + ", ".join(overlap)
        )
    seen.update(selected)
