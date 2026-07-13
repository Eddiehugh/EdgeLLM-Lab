"""Built-in optimizer parameter-group policies."""

from training.optimizers.policies.all import all_parameters
from training.optimizers.policies.base import ParamGroupPolicy
from training.optimizers.policies.decay_by_dimension import decay_by_dimension

__all__ = ["ParamGroupPolicy", "all_parameters", "decay_by_dimension"]
