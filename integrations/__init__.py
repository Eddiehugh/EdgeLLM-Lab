"""External project integration adapters."""

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import (
    INTEGRATION_REGISTRY,
    build_integration,
    integration_snapshot,
)

__all__ = [
    "INTEGRATION_REGISTRY",
    "IntegrationAdapter",
    "IntegrationInfo",
    "build_integration",
    "integration_snapshot",
]
