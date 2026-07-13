"""Replaceable optimizer algorithms, adapters, and grouping policies."""

from training.optimizers.registry import OPTIMIZER_REGISTRY, PARAM_GROUP_POLICY_REGISTRY

# Import built-ins so their registration decorators run before the public API is used.
from training.optimizers import adapters as adapters
from training.optimizers import policies as policies
from training.optimizers import reference as reference
from training.optimizers.api import (
    build_optimizer,
    resolve_optimizer_name,
    resolve_param_group_policy_name,
)
from training.optimizers.reference import ReferenceAdamW

__all__ = [
    "OPTIMIZER_REGISTRY",
    "PARAM_GROUP_POLICY_REGISTRY",
    "ReferenceAdamW",
    "build_optimizer",
    "resolve_optimizer_name",
    "resolve_param_group_policy_name",
]
