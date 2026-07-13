"""Single-group parameter policy."""

from __future__ import annotations

from collections.abc import Iterable

from core import Maturity, ProjectLevel
from training.optimizers.policies.base import NamedParameter, ParameterGroup, trainable_parameters
from training.optimizers.registry import PARAM_GROUP_POLICY_REGISTRY


@PARAM_GROUP_POLICY_REGISTRY.register(
    "all",
    "default",
    level=ProjectLevel.WORK,
    maturity=Maturity.PRODUCTION,
    capabilities=("parameter_grouping", "single_group"),
)
def all_parameters(
    named_parameters: Iterable[NamedParameter],
) -> list[ParameterGroup]:
    """Place every trainable parameter in one optimizer group."""

    parameters = [parameter for _, parameter in trainable_parameters(named_parameters)]
    return [{"params": parameters, "group_name": "all"}] if parameters else []
