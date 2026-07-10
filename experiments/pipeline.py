"""Config-driven execution of registered experiment stages."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from experiments.context import ExperimentContext, StageRecord, utc_now
from experiments.stage import ExperimentStage, build_stage


@dataclass(frozen=True)
class PipelineNode:
    name: str
    stage: ExperimentStage


class ExperimentPipeline:
    """Execute independent stages in a reproducible configured order."""

    def __init__(self, nodes: list[PipelineNode]):
        if not nodes:
            raise ValueError("Experiment pipeline requires at least one stage")
        self.nodes = nodes
        self._validate_order()

    def _validate_order(self) -> None:
        available: set[str] = set()
        for node in self.nodes:
            missing = [name for name in node.stage.requires if name not in available]
            if missing:
                raise ValueError(
                    f"Invalid pipeline order: stage '{node.name}' requires "
                    f"{', '.join(missing)} before they are provided"
                )
            available.update(node.stage.provides)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExperimentPipeline":
        pipeline_cfg = dict(config.get("pipeline", {}))
        stage_configs = list(pipeline_cfg.get("stages", []))
        nodes = []
        for stage_config in stage_configs:
            name, stage = build_stage(stage_config)
            nodes.append(PipelineNode(name=name, stage=stage))
        return cls(nodes)

    def run(self, context: ExperimentContext) -> None:
        for node in self.nodes:
            started_at = utc_now()
            start_time = time.perf_counter()
            try:
                missing = [
                    name for name in node.stage.requires if name not in context.objects
                ]
                if missing:
                    raise RuntimeError(
                        f"Stage '{node.name}' requires unavailable objects: "
                        f"{', '.join(missing)}"
                    )
                node.stage.run(context)
                missing_outputs = [
                    name for name in node.stage.provides if name not in context.objects
                ]
                if missing_outputs:
                    raise RuntimeError(
                        f"Stage '{node.name}' did not provide declared objects: "
                        f"{', '.join(missing_outputs)}"
                    )
            except Exception as exc:
                context.stages.append(
                    StageRecord(
                        name=node.name,
                        status="failed",
                        started_at=started_at,
                        finished_at=utc_now(),
                        duration_seconds=time.perf_counter() - start_time,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
            context.stages.append(
                StageRecord(
                    name=node.name,
                    status="completed",
                    started_at=started_at,
                    finished_at=utc_now(),
                    duration_seconds=time.perf_counter() - start_time,
                )
            )
