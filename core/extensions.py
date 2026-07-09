"""Dynamic extension loading for custom experiments."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any


def import_modules(module_names: Sequence[str]) -> list[str]:
    """Import dotted Python modules so their registry decorators run."""

    imported: list[str] = []
    for module_name in module_names:
        if not module_name:
            continue
        importlib.import_module(module_name)
        imported.append(module_name)
    return imported


def extension_modules_from_config(config: Mapping[str, Any]) -> list[str]:
    """Read extension module paths from common config locations."""

    modules: list[str] = []

    imports = config.get("imports", [])
    if isinstance(imports, str):
        modules.append(imports)
    else:
        modules.extend(imports)

    extensions = config.get("extensions", {})
    if isinstance(extensions, Mapping):
        configured = extensions.get("modules", [])
        if isinstance(configured, str):
            modules.append(configured)
        else:
            modules.extend(configured)

    return [str(module_name) for module_name in modules]


def load_extensions_from_config(config: Mapping[str, Any]) -> list[str]:
    """Import custom modules declared by an experiment config."""

    return import_modules(extension_modules_from_config(config))
