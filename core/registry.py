"""Small component registry used by all replaceable LLM parts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Map string names to component classes or factory functions."""

    def __init__(self, component_type: str):
        self.component_type = component_type
        self._items: dict[str, T] = {}

    def register(
        self,
        name: str,
        *aliases: str,
        override: bool = False,
    ) -> Callable[[T], T]:
        """Register a component under one or more stable config names."""

        def decorator(item: T) -> T:
            for raw_name in (name, *aliases):
                key = self._normalize(raw_name)
                if key in self._items and not override:
                    raise KeyError(f"{self.component_type} '{key}' is already registered")
                self._items[key] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        key = self._normalize(name)
        try:
            return self._items[key]
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
        return tuple(sorted(self._items))

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple((name, self._items[name]) for name in self.names())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._items

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()


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
