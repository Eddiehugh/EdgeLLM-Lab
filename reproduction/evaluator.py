"""Evaluate paper claims against metrics from one or more recipes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from reproduction.specs import ClaimSpec, ExpectationSpec


@dataclass(frozen=True)
class ExpectationResult:
    passed: bool
    metric: str
    recipe: str
    operator: str
    expected: Any
    observed: Any = None
    baseline: str | None = None
    baseline_value: Any = None
    mode: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    statement: str
    passed: bool | None
    expectations: tuple[ExpectationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def metric_value(metrics: Mapping[str, Any], path: str) -> Any:
    value: Any = metrics
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
            continue
        raise KeyError(path)
    return value


def _compare(
    observed: Any,
    operator: str,
    expected: Any,
    tolerance: float | None,
) -> bool:
    if operator == "==":
        return observed == expected
    if operator == "!=":
        return observed != expected
    if operator == "approx":
        return math.isclose(
            float(observed),
            float(expected),
            rel_tol=tolerance if tolerance is not None else 1e-5,
            abs_tol=tolerance if tolerance is not None else 1e-8,
        )
    if operator == "<":
        return observed < expected
    if operator == "<=":
        return observed <= expected
    if operator == ">":
        return observed > expected
    if operator == ">=":
        return observed >= expected
    raise ValueError(f"Unknown operator: {operator}")


def _comparison_value(candidate: Any, baseline: Any, mode: str) -> float:
    candidate_value = float(candidate)
    baseline_value = float(baseline)
    if mode == "delta":
        return candidate_value - baseline_value
    if baseline_value == 0:
        raise ZeroDivisionError("Baseline metric is zero")
    if mode == "ratio":
        return candidate_value / baseline_value
    if mode == "percent_change":
        return (candidate_value - baseline_value) / abs(baseline_value) * 100.0
    raise ValueError(f"Unknown comparison mode: {mode}")


def evaluate_expectation(
    expectation: ExpectationSpec,
    metrics_by_recipe: Mapping[str, Mapping[str, Any]],
) -> ExpectationResult:
    try:
        candidate_metrics = metrics_by_recipe[expectation.recipe]
        candidate = metric_value(candidate_metrics, expectation.metric)
        baseline_value = None
        observed = candidate
        if expectation.type == "comparison":
            baseline_metrics = metrics_by_recipe[expectation.baseline or ""]
            baseline_value = metric_value(baseline_metrics, expectation.metric)
            observed = _comparison_value(
                candidate, baseline_value, expectation.mode or "ratio"
            )
        passed = _compare(
            observed,
            expectation.operator,
            expectation.value,
            expectation.tolerance,
        )
        return ExpectationResult(
            passed=passed,
            metric=expectation.metric,
            recipe=expectation.recipe,
            operator=expectation.operator,
            expected=expectation.value,
            observed=observed,
            baseline=expectation.baseline,
            baseline_value=baseline_value,
            mode=expectation.mode,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return ExpectationResult(
            passed=False,
            metric=expectation.metric,
            recipe=expectation.recipe,
            operator=expectation.operator,
            expected=expectation.value,
            baseline=expectation.baseline,
            mode=expectation.mode,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_claims(
    claims: Sequence[ClaimSpec],
    metrics_by_recipe: Mapping[str, Mapping[str, Any]],
) -> list[ClaimResult]:
    results = []
    for claim in claims:
        if not claim.enabled:
            continue
        expectations = tuple(
            evaluate_expectation(expectation, metrics_by_recipe)
            for expectation in claim.expectations
        )
        passed = all(result.passed for result in expectations) if expectations else None
        results.append(
            ClaimResult(
                claim_id=claim.id,
                statement=claim.statement,
                passed=passed,
                expectations=expectations,
            )
        )
    return results
