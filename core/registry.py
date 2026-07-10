"""Small component registry used by all replaceable LLM parts."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from core.specs import ComponentSpec, Maturity, ProjectLevel


T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    """A registered target and its stable component specification."""

    target: T
    spec: ComponentSpec


class Registry(Generic[T]):
    """Map string names to component classes or factory functions."""

    def __init__(self, component_type: str):
        self.component_type = component_type
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        *aliases: str,
        override: bool = False,
        spec: ComponentSpec | None = None,
        description: str = "",
        level: ProjectLevel = ProjectLevel.LEARN,
        maturity: Maturity = Maturity.EXPERIMENTAL,
        capabilities: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
        provides: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        source: str | None = None,
    ) -> Callable[[T], T]:
        """Register a component and optional machine-readable metadata."""

        def decorator(item: T) -> T:
            canonical_name = self._normalize(name)
            normalized_aliases = tuple(self._normalize(alias) for alias in aliases)
            item_description = description or self._first_doc_line(item)
            if spec is None:
                resolved_level = (
                    level if isinstance(level, ProjectLevel) else ProjectLevel(level)
                )
                resolved_maturity = (
                    maturity if isinstance(maturity, Maturity) else Maturity(maturity)
                )
                resolved_requires = tuple(requires) or tuple(
                    getattr(item, "requires", ())
                )
                resolved_provides = tuple(provides) or tuple(
                    getattr(item, "provides", ())
                )
                item_spec = ComponentSpec(
                    name=canonical_name,
                    component_type=self.component_type,
                    description=item_description,
                    level=resolved_level,
                    maturity=resolved_maturity,
                    capabilities=tuple(capabilities),
                    requires=resolved_requires,
                    provides=resolved_provides,
                    tags=tuple(tags),
                    aliases=normalized_aliases,
                    source=source or getattr(item, "__module__", None),
                )
            else:
                item_spec = replace(
                    spec,
                    name=canonical_name,
                    component_type=self.component_type,
                    aliases=normalized_aliases or spec.aliases,
                )

            entry = RegistryEntry(target=item, spec=item_spec)
            keys = tuple(dict.fromkeys((canonical_name, *normalized_aliases)))
            conflicts = [key for key in keys if key in self._entries]
            if conflicts and not override:
                raise KeyError(
                    f"{self.component_type} names already registered: "
                    f"{', '.join(conflicts)}"
                )
            for key in keys:
                self._entries[key] = entry
            return item

        return decorator

    def get(self, name: str) -> T:
        key = self._normalize(name)
        try:
            return self._entries[key].target
        except KeyError as exc:
            available = ", ".join(self.names()) or "<empty>"
            raise KeyError(
                f"Unknown {self.component_type} '{key}'. Available: {available}"
            ) from exc

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        target = self.get(name)
        if not callable(target):
            raise TypeError(f"Registered {self.component_type} '{name}' is not callable")
        return target(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return every accepted config name, including aliases."""

        return tuple(sorted(self._entries))

    def canonical_names(self) -> tuple[str, ...]:
        """Return implementation names without aliases."""

        return tuple(sorted({entry.spec.name for entry in self._entries.values()}))

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple((name, self._entries[name].target) for name in self.names())

    def get_spec(self, name: str) -> ComponentSpec:
        key = self._normalize(name)
        try:
            return self._entries[key].spec
        except KeyError as exc:
            available = ", ".join(self.names()) or "<empty>"
            raise KeyError(
                f"Unknown {self.component_type} '{key}'. Available: {available}"
            ) from exc

    def describe(self, name: str) -> dict[str, Any]:
        """Return JSON-serializable metadata for a registered name."""

        return self.get_spec(name).to_dict()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return metadata keyed by canonical implementation name."""

        return {
            name: self.get_spec(name).to_dict()
            for name in self.canonical_names()
        }

    def require_capabilities(self, name: str, *capabilities: str) -> None:
        spec = self.get_spec(name)
        missing = [item for item in capabilities if item not in spec.capabilities]
        if missing:
            raise ValueError(
                f"{self.component_type} '{spec.name}' does not declare required "
                f"capabilities: {', '.join(missing)}"
            )

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._entries

    def __len__(self) -> int:
        return len(self.canonical_names())

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _first_doc_line(item: T) -> str:
        raw_doc = getattr(item, "__doc__", None) or ""
        doc = inspect.cleandoc(raw_doc)
        return doc.splitlines()[0] if doc else ""


def build_from_config(
    registry: Registry[Any],
    config: str | Mapping[str, Any] | None,
    *,
    default_type: str,
    type_key: str = "type",
    **base_kwargs: Any,
) -> Any:
    """Build a registered component from a string or mapping config."""

    if config is None:
        name = default_type
        kwargs: dict[str, Any] = {}
    elif isinstance(config, str):
        name = config
        kwargs = {}
    else:
        kwargs = dict(config)
        name = kwargs.pop(type_key, default_type)

    merged_kwargs = {**base_kwargs, **kwargs}
    return registry.build(str(name), **merged_kwargs)
