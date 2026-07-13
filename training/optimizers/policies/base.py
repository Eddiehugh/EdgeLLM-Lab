"""Contracts and shared helpers for optimizer parameter grouping."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch


NamedParameter = tuple[str, torch.nn.Parameter]
ParameterGroup = dict[str, object]


class ParamGroupPolicy(Protocol):
    """Turn named model parameters into PyTorch optimizer groups."""

    def __call__(
        self,
        named_parameters: Iterable[NamedParameter],
        **kwargs: object,
    ) -> list[ParameterGroup]: ...


def trainable_parameters(
    named_parameters: Iterable[NamedParameter],
) -> list[NamedParameter]:
    """Collect trainable parameters once, preserving their stable names."""

    result: list[NamedParameter] = []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append((name, parameter))
    return result
