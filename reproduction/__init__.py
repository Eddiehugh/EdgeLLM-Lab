"""Paper reproduction workspace, study runner, and claim evaluation."""

from reproduction.evaluator import (
    ClaimResult,
    ExpectationResult,
    evaluate_claims,
    evaluate_expectation,
)
from reproduction.manager import PaperStudyManager
from reproduction.specs import (
    PAPER_SCHEMA_VERSION,
    ClaimSpec,
    ExpectationSpec,
    PaperConfigError,
    PaperSpec,
    RecipeSpec,
    SuiteSpec,
)
from reproduction.workspace import PaperWorkspace

__all__ = [
    "PAPER_SCHEMA_VERSION",
    "ClaimResult",
    "ClaimSpec",
    "ExpectationResult",
    "ExpectationSpec",
    "PaperConfigError",
    "PaperSpec",
    "PaperStudyManager",
    "PaperWorkspace",
    "RecipeSpec",
    "SuiteSpec",
    "evaluate_claims",
    "evaluate_expectation",
]
