"""Dynamic extension loading for custom experiments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
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


def extension_paths_from_config(config: Mapping[str, Any]) -> list[str]:
    """Read standalone extension file paths from config."""

    paths: list[str] = []
    configured_paths = config.get("extension_paths", [])
    if isinstance(configured_paths, str):
        paths.append(configured_paths)
    else:
        paths.extend(configured_paths)

    extensions = config.get("extensions", {})
    if isinstance(extensions, Mapping):
        configured_paths = extensions.get("paths", [])
        if isinstance(configured_paths, str):
            paths.append(configured_paths)
        else:
            paths.extend(configured_paths)
    return [str(path) for path in paths]


def import_extension_paths(paths: Sequence[str]) -> list[str]:
    """Load standalone Python extension files under deterministic module names."""

    imported = []
    for configured_path in paths:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Extension file does not exist: {path}")
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        module_name = f"_edgellm_extension_{path.stem}_{digest}"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load extension file: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        imported.append(str(path))
    return imported


def load_extensions_from_config(config: Mapping[str, Any]) -> list[str]:
    """Import custom modules declared by an experiment config."""

    imported = import_modules(extension_modules_from_config(config))
    imported.extend(import_extension_paths(extension_paths_from_config(config)))
    return imported
