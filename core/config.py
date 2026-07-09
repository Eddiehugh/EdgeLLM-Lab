"""Configuration loading and normalization helpers."""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on local environment
    yaml = None


ConfigDict = dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    """Load a JSON or YAML config file."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files")
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported config suffix: {config_path.suffix}")

    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write config as YAML when possible, otherwise JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in {".yaml", ".yml"} and yaml is not None:
        text = yaml.safe_dump(dict(config), sort_keys=False)
    else:
        text = json.dumps(config, indent=2, sort_keys=False)
    output_path.write_text(text + "\n", encoding="utf-8")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Merge nested dictionaries without mutating either input."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_scalar(value: str) -> Any:
    """Parse CLI override values into Python scalars when possible."""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def dotlist_to_dict(overrides: Sequence[str]) -> ConfigDict:
    """Convert ['a.b=1'] style overrides into a nested dict."""

    result: ConfigDict = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have key=value form: {override}")
        key, raw_value = override.split("=", 1)
        if not key:
            raise ValueError(f"Override key is empty: {override}")

        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path conflicts with scalar value: {key}")
        cursor[parts[-1]] = parse_scalar(raw_value)
    return result


def with_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> ConfigDict:
    """Apply CLI dotlist overrides to a config mapping."""

    if not overrides:
        return copy.deepcopy(dict(config))
    return deep_merge(config, dotlist_to_dict(overrides))


def component_config(
    section: Mapping[str, Any],
    *,
    type_keys: Sequence[str] = ("type", "name"),
    default_type: str | None = None,
    exclude_keys: Sequence[str] = (),
) -> tuple[str | dict[str, Any], dict[str, Any]]:
    """Split a config section into a component selector and constructor kwargs."""

    kwargs = dict(section)
    for key in exclude_keys:
        kwargs.pop(key, None)

    selected_key = next((key for key in type_keys if key in kwargs), None)
    if selected_key is None:
        if default_type is None:
            raise KeyError(f"Missing component type key. Expected one of: {type_keys}")
        return default_type, kwargs

    component_type = kwargs.pop(selected_key)
    return str(component_type), kwargs
