"""Validated specifications for paper reproduction studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PAPER_SCHEMA_VERSION = 1
EXPECTATION_TYPES = {"absolute", "comparison"}
OPERATORS = {"==", "!=", "<", "<=", ">", ">=", "approx"}
COMPARISON_MODES = {"ratio", "delta", "percent_change"}


class PaperConfigError(ValueError):
    """Raised when a paper manifest or recipe is invalid."""


def _require_text(data: Mapping[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PaperConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExpectationSpec:
    metric: str
    recipe: str
    operator: str
    value: Any
    type: str = "absolute"
    baseline: str | None = None
    mode: str | None = None
    tolerance: float | None = None
    note: str | None = None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], location: str
    ) -> "ExpectationSpec":
        values = dict(data)
        expectation_type = str(values.get("type", "absolute"))
        if expectation_type not in EXPECTATION_TYPES:
            raise PaperConfigError(
                f"{location}.type must be one of: {', '.join(EXPECTATION_TYPES)}"
            )
        operator = str(values.get("operator", "=="))
        if operator not in OPERATORS:
            raise PaperConfigError(
                f"{location}.operator must be one of: {', '.join(OPERATORS)}"
            )
        if "value" not in values:
            raise PaperConfigError(f"{location}.value is required")
        baseline = values.get("baseline")
        mode = values.get("mode")
        if expectation_type == "comparison":
            if not isinstance(baseline, str) or not baseline:
                raise PaperConfigError(
                    f"{location}.baseline is required for comparison expectations"
                )
            mode = str(mode or "ratio")
            if mode not in COMPARISON_MODES:
                raise PaperConfigError(
                    f"{location}.mode must be one of: "
                    f"{', '.join(COMPARISON_MODES)}"
                )
        tolerance = values.get("tolerance")
        if tolerance is not None:
            try:
                tolerance = float(tolerance)
            except (TypeError, ValueError) as exc:
                raise PaperConfigError(f"{location}.tolerance must be numeric") from exc
            if tolerance < 0:
                raise PaperConfigError(f"{location}.tolerance must be non-negative")
        return cls(
            metric=_require_text(values, "metric", location),
            recipe=_require_text(values, "recipe", location),
            operator=operator,
            value=values["value"],
            type=expectation_type,
            baseline=str(baseline) if baseline is not None else None,
            mode=str(mode) if mode is not None else None,
            tolerance=tolerance,
            note=str(values["note"]) if values.get("note") else None,
        )


@dataclass(frozen=True)
class ClaimSpec:
    id: str
    statement: str
    expectations: tuple[ExpectationSpec, ...] = ()
    enabled: bool = True
    source: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int) -> "ClaimSpec":
        location = f"claims[{index}]"
        expectations_data = data.get("expectations", [])
        if not isinstance(expectations_data, list):
            raise PaperConfigError(f"{location}.expectations must be a list")
        expectations = tuple(
            ExpectationSpec.from_dict(value, f"{location}.expectations[{item_index}]")
            for item_index, value in enumerate(expectations_data)
            if isinstance(value, Mapping)
        )
        if len(expectations) != len(expectations_data):
            raise PaperConfigError(
                f"Every item in {location}.expectations must be a mapping"
            )
        return cls(
            id=_require_text(data, "id", location),
            statement=_require_text(data, "statement", location),
            expectations=expectations,
            enabled=bool(data.get("enabled", True)),
            source=str(data["source"]) if data.get("source") else None,
        )


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    recipes: tuple[str, ...]
    claims: tuple[str, ...] = ()
    strategy: str = "sequential"

    @classmethod
    def from_dict(
        cls, name: str, data: Mapping[str, Any]
    ) -> "SuiteSpec":
        recipes = data.get("recipes")
        if not isinstance(recipes, list) or not recipes:
            raise PaperConfigError(f"suites.{name}.recipes must be a non-empty list")
        strategy = str(data.get("strategy", "sequential"))
        if strategy not in {"sequential", "parallel"}:
            raise PaperConfigError(
                f"suites.{name}.strategy must be sequential or parallel"
            )
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            raise PaperConfigError(f"suites.{name}.claims must be a list")
        return cls(
            name=name,
            recipes=tuple(str(value) for value in recipes),
            claims=tuple(str(value) for value in claims),
            strategy=strategy,
        )


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    raw: dict[str, Any]
    base_config: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], location: str) -> "RecipeSpec":
        version = data.get("schema_version", PAPER_SCHEMA_VERSION)
        if version != PAPER_SCHEMA_VERSION:
            raise PaperConfigError(
                f"{location}.schema_version must be {PAPER_SCHEMA_VERSION}"
            )
        base_config = data.get("base_config")
        inline_config = data.get("config", {})
        overrides = data.get("overrides", {})
        execution = data.get("execution", {})
        for key, value in (
            ("config", inline_config),
            ("overrides", overrides),
            ("execution", execution),
        ):
            if not isinstance(value, Mapping):
                raise PaperConfigError(f"{location}.{key} must be a mapping")
        if base_config is None and not inline_config:
            raise PaperConfigError(
                f"{location} requires base_config or a non-empty config mapping"
            )
        return cls(
            name=_require_text(data, "name", location),
            raw=dict(data),
            base_config=str(base_config) if base_config is not None else None,
            config=dict(inline_config),
            overrides=dict(overrides),
            execution=dict(execution),
            description=str(data["description"]) if data.get("description") else None,
        )


@dataclass(frozen=True)
class PaperSpec:
    id: str
    title: str
    manifest: dict[str, Any]
    claims: tuple[ClaimSpec, ...]
    suites: dict[str, SuiteSpec]
    authors: tuple[str, ...] = ()
    url: str | None = None
    year: int | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperSpec":
        version = data.get("schema_version")
        if version != PAPER_SCHEMA_VERSION:
            raise PaperConfigError(
                f"schema_version must be {PAPER_SCHEMA_VERSION}; received {version!r}"
            )
        paper = data.get("paper")
        if not isinstance(paper, Mapping):
            raise PaperConfigError("paper must be a mapping")
        claims_data = data.get("claims", [])
        if not isinstance(claims_data, list):
            raise PaperConfigError("claims must be a list")
        claims = tuple(
            ClaimSpec.from_dict(value, index)
            for index, value in enumerate(claims_data)
            if isinstance(value, Mapping)
        )
        if len(claims) != len(claims_data):
            raise PaperConfigError("Every item in claims must be a mapping")
        claim_ids = [claim.id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise PaperConfigError("Claim ids must be unique")

        suites_data = data.get("suites", {})
        if not isinstance(suites_data, Mapping) or not suites_data:
            raise PaperConfigError("suites must be a non-empty mapping")
        suites = {
            str(name): SuiteSpec.from_dict(str(name), value)
            for name, value in suites_data.items()
            if isinstance(value, Mapping)
        }
        if len(suites) != len(suites_data):
            raise PaperConfigError("Every suite must be a mapping")
        known_claims = set(claim_ids)
        for suite in suites.values():
            unknown = set(suite.claims) - known_claims
            if unknown:
                raise PaperConfigError(
                    f"Suite '{suite.name}' references unknown claims: "
                    f"{', '.join(sorted(unknown))}"
                )

        authors = paper.get("authors", [])
        tags = paper.get("tags", [])
        if not isinstance(authors, list) or not isinstance(tags, list):
            raise PaperConfigError("paper.authors and paper.tags must be lists")
        year = paper.get("year")
        if year is not None:
            try:
                year = int(year)
            except (TypeError, ValueError) as exc:
                raise PaperConfigError("paper.year must be an integer") from exc
        return cls(
            id=_require_text(paper, "id", "paper"),
            title=_require_text(paper, "title", "paper"),
            manifest=dict(data),
            claims=claims,
            suites=suites,
            authors=tuple(str(value) for value in authors),
            url=str(paper["url"]) if paper.get("url") else None,
            year=year,
            tags=tuple(str(value) for value in tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def claim(self, claim_id: str) -> ClaimSpec:
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        raise KeyError(f"Unknown claim '{claim_id}' for paper '{self.id}'")
