"""Markdown reports for paper reproduction studies."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from reproduction.evaluator import ClaimResult
from reproduction.specs import PaperSpec


def _status(value: bool | None) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "NOT ASSESSED"


def build_study_report(
    paper: PaperSpec,
    study: Mapping[str, Any],
    claims: Sequence[ClaimResult],
) -> str:
    lines = [
        f"# Reproduction: {paper.title}",
        "",
        "## Study",
        "",
        f"- Paper ID: `{paper.id}`",
        f"- Study ID: `{study['study_id']}`",
        f"- Suite: `{study['suite']}`",
        f"- Status: `{study['status']}`",
    ]
    if paper.url:
        lines.append(f"- Paper: {paper.url}")
    lines.extend(
        [
            "",
            "## Recipes",
            "",
            "| Recipe | Job | Executor | State | Artifact |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for recipe, job in study.get("jobs", {}).items():
        lines.append(
            f"| `{recipe}` | `{job.get('job_id', '-')}` | "
            f"`{job.get('executor_type', '-')}` | `{job.get('state', '-')}` | "
            f"{job.get('artifact_uri') or '-'} |"
        )

    lines.extend(["", "## Claim Assessment", ""])
    if not claims:
        lines.append("No enabled claims were selected for this suite.")
    for claim in claims:
        lines.extend(
            [
                f"### {_status(claim.passed)}: {claim.claim_id}",
                "",
                claim.statement,
                "",
                "| Metric | Recipe | Baseline | Observed | Rule | Result |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        if not claim.expectations:
            lines.append("| - | - | - | - | - | NOT ASSESSED |")
        for result in claim.expectations:
            observed = result.error or result.observed
            lines.append(
                f"| `{result.metric}` | `{result.recipe}` | "
                f"`{result.baseline or '-'}` | {observed} | "
                f"`{result.operator} {result.expected}` | "
                f"{'PASS' if result.passed else 'FAIL'} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A passing harness does not prove the paper. Record differences in data, "
            "scale, hardware, kernels, hyperparameters, and evaluation protocol before "
            "claiming reproduction success.",
            "",
        ]
    )
    return "\n".join(lines)
