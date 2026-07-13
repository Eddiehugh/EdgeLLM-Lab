"""Registries for optimizer implementations and parameter-group policies."""

from __future__ import annotations

from core.registry import Registry


OPTIMIZER_REGISTRY = Registry("optimizer")
PARAM_GROUP_POLICY_REGISTRY = Registry("optimizer_param_group_policy")
