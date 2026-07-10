"""Core framework utilities."""

from core.config import (
    component_config,
    deep_merge,
    dotlist_to_dict,
    load_config,
    save_config,
    with_overrides,
)
from core.extensions import load_extensions_from_config
from core.provenance import capture_environment
from core.registry import Registry, RegistryEntry, build_from_config
from core.runtime import count_parameters, model_size_bytes, resolve_device, set_seed, timed
from core.specs import ComponentSpec, Maturity, ProjectLevel

__all__ = [
    "Registry",
    "RegistryEntry",
    "ComponentSpec",
    "Maturity",
    "ProjectLevel",
    "build_from_config",
    "capture_environment",
    "component_config",
    "count_parameters",
    "deep_merge",
    "dotlist_to_dict",
    "load_config",
    "load_extensions_from_config",
    "model_size_bytes",
    "resolve_device",
    "save_config",
    "set_seed",
    "timed",
    "with_overrides",
]
