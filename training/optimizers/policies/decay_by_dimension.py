"""Weight-decay grouping based on parameter dimensionality."""

from __future__ import annotations

from collections.abc import Iterable

from core import Maturity, ProjectLevel
from training.optimizers.policies.base import NamedParameter, ParameterGroup, trainable_parameters
from training.optimizers.registry import PARAM_GROUP_POLICY_REGISTRY


@PARAM_GROUP_POLICY_REGISTRY.register(
    "decay_by_dimension",
    "matrix_decay",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("parameter_grouping", "weight_decay_partition"),
)
def decay_by_dimension(
    named_parameters: Iterable[NamedParameter],
    min_decay_ndim: int = 2,
) -> list[ParameterGroup]:
    """Apply weight decay to matrices and exclude vectors and scalars."""

    if min_decay_ndim < 1:
        raise ValueError("min_decay_ndim must be at least 1")

    decay = []
    no_decay = []
    for _, parameter in trainable_parameters(named_parameters):
        target = decay if parameter.ndim >= min_decay_ndim else no_decay
        target.append(parameter)

    groups: list[ParameterGroup] = []
    if decay:
        groups.append({"params": decay, "group_name": "decay"})
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "group_name": "no_decay",
                "weight_decay": 0.0,
            }
        )
    return groups
