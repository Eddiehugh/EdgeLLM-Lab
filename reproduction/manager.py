"""Run and assess reproducible paper recipe suites."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from core.config import deep_merge, with_overrides
from execution import JobRecord, JobState, RunManager
from reproduction.evaluator import ClaimResult, evaluate_claims
from reproduction.report import build_study_report
from reproduction.specs import ClaimSpec, PaperConfigError
from reproduction.workspace import PaperWorkspace


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperStudyManager:
    def __init__(
        self,
        workspace: PaperWorkspace | None = None,
        run_manager: RunManager | None = None,
        state_root: str | Path | None = None,
    ):
        self.workspace = workspace or PaperWorkspace()
        self.run_manager = run_manager or RunManager(
            project_root=self.workspace.project_root
        )
        root = Path(
            state_root
            or self.workspace.project_root / ".edgellm" / "paper-studies"
        ).expanduser()
        self.state_root = root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _study_directory(self, paper_id: str, study_id: str) -> Path:
        return self.state_root / paper_id / study_id

    def _save(self, study: dict[str, Any]) -> Path:
        directory = self._study_directory(study["paper_id"], study["study_id"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "study.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(study, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load_study(self, paper_id: str, study_id: str) -> dict[str, Any]:
        path = self._study_directory(paper_id, study_id) / "study.json"
        if not path.exists():
            raise KeyError(f"Unknown study '{study_id}' for paper '{paper_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _job_summary(record: JobRecord) -> dict[str, Any]:
        return {
            "job_id": record.job_id,
            "executor_type": record.executor_type,
            "state": record.state.value,
            "artifact_uri": record.artifact_uri,
            "url": record.url,
            "message": record.message,
        }

    def _recipe_config(
        self,
        paper_id: str,
        recipe: str,
        executor: str | None,
        overrides: Sequence[str],
    ) -> dict[str, Any]:
        config = self.workspace.resolve_recipe(paper_id, recipe)
        if executor:
            config = deep_merge(
                config,
                {"execution": {"executor": {"type": executor}}},
            )
        return with_overrides(config, overrides)

    def run_study(
        self,
        paper_id: str,
        suite_name: str = "smoke",
        *,
        executor: str | None = None,
        overrides: Sequence[str] = (),
        detach: bool = False,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.workspace.validate(paper_id)
        paper = self.workspace.load(paper_id)
        try:
            suite = paper.suites[suite_name]
        except KeyError as exc:
            raise PaperConfigError(
                f"Unknown suite '{suite_name}'. Available: {', '.join(paper.suites)}"
            ) from exc
        study_id = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{suite.name}-{uuid.uuid4().hex[:6]}"
        )
        study: dict[str, Any] = {
            "study_version": 1,
            "study_id": study_id,
            "paper_id": paper.id,
            "title": paper.title,
            "suite": suite.name,
            "strategy": suite.strategy,
            "status": "submitting",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "jobs": {},
            "metrics": {},
            "claims": [],
        }
        self._save(study)

        try:
            if detach or suite.strategy == "parallel":
                for recipe in suite.recipes:
                    record = self.run_manager.submit(
                        self._recipe_config(paper.id, recipe, executor, overrides)
                    )
                    study["jobs"][recipe] = self._job_summary(record)
                    study["updated_at"] = utc_now()
                    self._save(study)
            else:
                for recipe in suite.recipes:
                    record = self.run_manager.submit(
                        self._recipe_config(paper.id, recipe, executor, overrides)
                    )
                    study["jobs"][recipe] = self._job_summary(record)
                    study["updated_at"] = utc_now()
                    self._save(study)
                    completed = self.run_manager.wait(
                        record.job_id,
                        poll_interval=poll_interval,
                        timeout=timeout,
                    )
                    study["jobs"][recipe] = self._job_summary(completed)
                    study["updated_at"] = utc_now()
                    self._save(study)
        except Exception as exc:
            study["status"] = "orchestration_failed"
            study["error"] = f"{type(exc).__name__}: {exc}"
            study["updated_at"] = utc_now()
            self._save(study)
            raise

        if detach:
            study["status"] = "submitted"
            study["updated_at"] = utc_now()
            self._save(study)
            return study
        return self.assess_study(
            paper.id,
            study_id,
            wait=True,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def _selected_claims(self, paper_id: str, suite_name: str) -> list[ClaimSpec]:
        paper = self.workspace.load(paper_id)
        suite = paper.suites[suite_name]
        if suite.claims:
            return [paper.claim(claim_id) for claim_id in suite.claims]
        return list(paper.claims)

    def _load_metrics(
        self, paper_id: str, study_id: str, recipe: str, job_id: str
    ) -> dict[str, Any]:
        artifact_path = (
            self._study_directory(paper_id, study_id) / "artifacts" / recipe
        )
        if not artifact_path.exists():
            self.run_manager.fetch(job_id, artifact_path)
        metrics_path = artifact_path / "metrics.json"
        if not metrics_path.exists():
            candidates = list(artifact_path.rglob("metrics.json"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one metrics.json under {artifact_path}; "
                    f"found {len(candidates)}"
                )
            metrics_path = candidates[0]
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def assess_study(
        self,
        paper_id: str,
        study_id: str,
        *,
        wait: bool = False,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        study = self.load_study(paper_id, study_id)
        for recipe, job in study["jobs"].items():
            if wait:
                record = self.run_manager.wait(
                    job["job_id"],
                    poll_interval=poll_interval,
                    timeout=timeout,
                )
            else:
                record = self.run_manager.status(job["job_id"])
            study["jobs"][recipe] = self._job_summary(record)
            if record.state == JobState.COMPLETED:
                try:
                    study["metrics"][recipe] = self._load_metrics(
                        paper_id, study_id, recipe, record.job_id
                    )
                except Exception as exc:
                    study["jobs"][recipe]["metrics_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

        states = {job["state"] for job in study["jobs"].values()}
        terminal = states <= {
            JobState.COMPLETED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        }
        selected_claims = self._selected_claims(paper_id, study["suite"])
        claim_results: list[ClaimResult] = []
        metrics_failed = any(
            "metrics_error" in job for job in study["jobs"].values()
        )
        if terminal:
            claim_results = evaluate_claims(selected_claims, study["metrics"])
            study["claims"] = [result.to_dict() for result in claim_results]
            jobs_passed = states == {JobState.COMPLETED.value}
            claims_passed = all(
                result.passed is not False for result in claim_results
            )
            study["status"] = (
                "completed"
                if jobs_passed and claims_passed and not metrics_failed
                else "failed"
            )
            study["finished_at"] = utc_now()
        elif JobState.PREPARED.value in states:
            study["status"] = "prepared"
        else:
            study["status"] = "running"
        study["updated_at"] = utc_now()
        self._save(study)

        if terminal:
            paper = self.workspace.load(paper_id)
            report = build_study_report(paper, study, claim_results)
            report_path = self._study_directory(paper_id, study_id) / "report.md"
            report_path.write_text(report, encoding="utf-8")
            study["report_path"] = str(report_path)
            self._save(study)
        return study
