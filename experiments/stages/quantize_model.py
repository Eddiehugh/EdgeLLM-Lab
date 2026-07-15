"""Config-driven model quantization stage."""

from __future__ import annotations

from compression import (
    ModuleSelector,
    QUANTIZER_REGISTRY,
    build_quantizer,
    quantize_linear_modules,
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
    "quantize_model",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "compression", "quantization", "weight_only"),
)
class QuantizeModelStage(ExperimentStage):
    """Replace selected Linear layers with reference weight-only modules."""

    requires = ("model",)
    provides = ("quantization_report",)

    def run(self, context: ExperimentContext) -> None:
        compression_cfg = dict(context.config.get("compression", {}))
        configured = compression_cfg.get("quantization")
        if configured is None:
            raise ValueError("quantize_model stage requires compression.quantization")
        model = context.require("model")
        reports = []
        seen_modules: set[str] = set()
        seen_labels: set[str] = set()
        configured_passes = compression_passes(
            configured,
            "compression.quantization",
        )
        for index, configured_pass in enumerate(configured_passes):
            pass_id, quantizer_cfg, transform_cfg = split_compression_pass(
                configured_pass,
                transform_fields=(
                    "selector",
                    "inplace",
                    "allow_empty",
                    "allow_shared_weights",
                ),
            )
            pass_label = unique_pass_label(pass_id, index, seen_labels)
            selector = ModuleSelector.from_config(transform_cfg.get("selector"))
            quantizer = build_quantizer(quantizer_cfg)
            model, pass_report = quantize_linear_modules(
                model,
                quantizer,
                selector=selector,
                inplace=bool(transform_cfg.get("inplace", True)),
                allow_shared_weights=bool(
                    transform_cfg.get("allow_shared_weights", False)
                ),
            )
            if not pass_report.records and not bool(
                transform_cfg.get("allow_empty", False)
            ):
                raise ValueError(
                    f"Quantization pass '{pass_label}' did not match any Linear modules"
                )
            reject_overlapping_modules(
                seen_modules,
                pass_report,
                pass_label=pass_label,
            )
            reports.append(pass_report)
            role = (
                "quantizer"
                if len(configured_passes) == 1
                else f"quantizer.{pass_label}"
            )
            context.track_component(
                role,
                QUANTIZER_REGISTRY,
                selected_name(quantizer_cfg, "int8"),
            )

        report = combined_report("quantization", reports)
        context.provide("model", model)
        context.provide("quantization_report", report)
        context.metrics["quantization"] = report.to_dict()
        context.metrics["quantization_pass_count"] = len(reports)
        context.metrics["quantized_module_count"] = len(report.records)
        context.metrics["quantized_weight_compression_ratio"] = report.compression_ratio
        context.metrics["quantized_model_compression_ratio"] = (
            report.model_compression_ratio
        )
