"""Compatibility facade for the structured optimizer package.

New code should import from :mod:`training.optimizers`. This module preserves
the original public path for existing experiments and external extensions.
"""

from training.optimizers import (
    OPTIMIZER_REGISTRY,
    PARAM_GROUP_POLICY_REGISTRY,
    ReferenceAdamW,
    build_optimizer,
    resolve_optimizer_name,
    resolve_param_group_policy_name,
)
from training.optimizers.adapters import (
    build_torch_adamw as build_adamw,
    build_torch_sgd as build_sgd,
)

__all__ = [
    "OPTIMIZER_REGISTRY",
    "PARAM_GROUP_POLICY_REGISTRY",
    "ReferenceAdamW",
    "build_adamw",
    "build_optimizer",
    "build_sgd",
    "resolve_optimizer_name",
    "resolve_param_group_policy_name",
]
