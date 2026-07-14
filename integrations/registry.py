"""Registry for external project integration adapters."""

from __future__ import annotations

from typing import Any

from core.registry import Registry, build_from_config
from integrations.base import IntegrationAdapter


INTEGRATION_REGISTRY = Registry[type[IntegrationAdapter]]("integration")
_BUILTINS_LOADED = False


def _load_builtin_integrations() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    import integrations.llama_cpp.adapter  # noqa: F401
    import integrations.mobilellm.adapter  # noqa: F401
    import integrations.nanochat.adapter  # noqa: F401
    import integrations.smollm.adapter  # noqa: F401
    import integrations.tinyllama.adapter  # noqa: F401


def build_integration(integration_type: str | dict, **kwargs: Any) -> IntegrationAdapter:
    """Build an integration adapter by name."""

    _load_builtin_integrations()
    return build_from_config(
        INTEGRATION_REGISTRY,
        integration_type,
        default_type="nanochat",
        **kwargs,
    )


def integration_snapshot(external_root: str = "external_projects") -> dict[str, dict[str, Any]]:
    """Return metadata for all known built-in integrations."""

    _load_builtin_integrations()
    snapshot: dict[str, dict[str, Any]] = {}
    for name in INTEGRATION_REGISTRY.names():
        adapter = build_integration(name, external_root=external_root)
        snapshot[adapter.name] = adapter.validate()
    return snapshot
