"""Config-driven model pruning stage."""

from __future__ import annotations

from compression import (
    ModuleSelector,
    PRUNER_REGISTRY,
    build_pruner,
    prune_linear_modules,
)
from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage
from experiments.stages.common import selected_name
from experiments.stages.compression_common import (
    combined_report,
    compression_passes,
    reject_overlapping_modules,
    split_compression_pass,
    unique_pass_label,
)


@STAGE_REGISTRY.register(
    "prune_model",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "compression", "pruning"),
)
class PruneModelStage(ExperimentStage):
    """Apply a registered pruning method to selected model modules."""

    requires = ("model",)
    provides = ("pruning_report",)

    def run(self, context: ExperimentContext) -> None:
        compression_cfg = dict(context.config.get("compression", {}))
        configured = compression_cfg.get("pruning")
        if configured is None:
            raise ValueError("prune_model stage requires compression.pruning")
        model = context.require("model")
        reports = []
        seen_modules: set[str] = set()
        seen_labels: set[str] = set()
        configured_passes = compression_passes(
            configured,
            "compression.pruning",
        )
        for index, configured_pass in enumerate(configured_passes):
            pass_id, pruner_cfg, transform_cfg = split_compression_pass(
                configured_pass,
                transform_fields=(
                    "selector",
                    "inplace",
                    "enforce_mask",
                    "allow_empty",
                    "allow_shared_weights",
                ),
            )
            pass_label = unique_pass_label(pass_id, index, seen_labels)
            selector = ModuleSelector.from_config(transform_cfg.get("selector"))
            pruner = build_pruner(pruner_cfg)
            model, pass_report = prune_linear_modules(
                model,
                pruner,
                selector=selector,
                inplace=bool(transform_cfg.get("inplace", True)),
                enforce_mask=bool(transform_cfg.get("enforce_mask", False)),
                allow_shared_weights=bool(
                    transform_cfg.get("allow_shared_weights", False)
                ),
            )
            if not pass_report.records and not bool(
                transform_cfg.get("allow_empty", False)
            ):
                raise ValueError(
                    f"Pruning pass '{pass_label}' did not match any Linear modules"
                )
            reject_overlapping_modules(
                seen_modules,
                pass_report,
                pass_label=pass_label,
            )
            reports.append(pass_report)
            role = "pruner" if len(configured_passes) == 1 else f"pruner.{pass_label}"
            context.track_component(
                role,
                PRUNER_REGISTRY,
                selected_name(pruner_cfg, "magnitude"),
            )

        report = combined_report("pruning", reports)
        context.provide("model", model)
        context.provide("pruning_report", report)
        context.metrics["pruning"] = report.to_dict()
        context.metrics["pruning_pass_count"] = len(reports)
        context.metrics["pruned_module_count"] = len(report.records)
        context.metrics["selected_weight_sparsity"] = report.affected_fraction
        context.metrics["pruned_model_compression_ratio"] = (
            report.model_compression_ratio
        )
