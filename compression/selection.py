"""Architecture-neutral module selection for compression transforms."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


def _tuple_of_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"compression selector {field_name} must contain strings")
    return tuple(value)


@dataclass(frozen=True)
class ModuleSelector:
    """Select modules by path patterns and optional model-defined scopes."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "ModuleSelector":
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("compression selector must be a mapping")
        values = dict(config or {})
        include = _tuple_of_strings(values.pop("include", ("*",)), "include")
        exclude = _tuple_of_strings(values.pop("exclude", ()), "exclude")
        scopes = _tuple_of_strings(values.pop("scopes", ()), "scopes")
        if values:
            raise ValueError(
                "Unknown compression selector fields: " + ", ".join(sorted(values))
            )
        return cls(include=include or ("*",), exclude=exclude, scopes=scopes)

    def _scope_prefixes(self, model: nn.Module) -> tuple[str, ...]:
        if not self.scopes:
            return ()
        provider = getattr(model, "compression_scopes", None)
        scopes = provider() if callable(provider) else provider
        if not isinstance(scopes, Mapping):
            raise ValueError(
                "Compression scopes were requested, but the model does not expose "
                "a compression_scopes mapping"
            )
        prefixes: list[str] = []
        for scope in self.scopes:
            if scope not in scopes:
                raise KeyError(
                    f"Unknown model compression scope '{scope}'. Available: "
                    + ", ".join(sorted(str(name) for name in scopes))
                )
            prefixes.extend(_tuple_of_strings(scopes[scope], f"scope '{scope}'"))
        return tuple(dict.fromkeys(prefixes))

    def select(
        self,
        model: nn.Module,
        module_type: type[nn.Module],
    ) -> list[tuple[str, nn.Module]]:
        prefixes = self._scope_prefixes(model)
        selected = []
        for name, module in model.named_modules():
            if not name or not isinstance(module, module_type):
                continue
            if prefixes and not any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in prefixes
            ):
                continue
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self.include):
                continue
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.exclude):
                continue
            selected.append((name, module))
        return selected


def replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace one named child while preserving the surrounding model object."""

    parent_path, _, child_name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def validate_weight_sharing(
    model: nn.Module,
    selected: list[tuple[str, nn.Module]],
    *,
    allow_shared_weights: bool,
) -> dict[str, tuple[str, ...]]:
    """Reject ambiguous transforms of tied parameters unless explicitly allowed."""

    owners: dict[int, list[str]] = {}
    for module_name, module in model.named_modules():
        weight = module._parameters.get("weight")
        if weight is None:
            continue
        path = f"{module_name}.weight" if module_name else "weight"
        owners.setdefault(id(weight), []).append(path)

    selected_names = {name for name, _ in selected}
    sharing: dict[str, tuple[str, ...]] = {}
    for name, module in selected:
        weight = module._parameters.get("weight")
        paths = tuple(owners.get(id(weight), ())) if weight is not None else ()
        if len(paths) <= 1:
            continue
        selected_owners = {
            path.rpartition(".")[0]
            for path in paths
            if path.rpartition(".")[0] in selected_names
        }
        if len(selected_owners) > 1:
            raise ValueError(
                f"Selected module '{name}' shares its weight with another selected "
                "module; transforming the same parameter twice is unsupported"
            )
        if not allow_shared_weights:
            raise ValueError(
                f"Selected module '{name}' uses tied weight {paths}. Exclude it or "
                "set allow_shared_weights=true to acknowledge that the transform "
                "may break or affect the tie."
            )
        sharing[name] = paths
    return sharing
