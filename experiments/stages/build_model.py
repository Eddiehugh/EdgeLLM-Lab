"""Build the configured model without coupling the pipeline to a model family."""

from __future__ import annotations

from core import component_config
from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage
from experiments.stages.common import selected_name
from models import MODEL_REGISTRY, build_model
from modules import (
    ATTENTION_REGISTRY,
    BLOCK_REGISTRY,
    MLP_REGISTRY,
    MULTIMODAL_FUSION_REGISTRY,
    MULTIMODAL_PROJECTOR_REGISTRY,
    MULTIMODAL_RESAMPLER_REGISTRY,
    NORM_REGISTRY,
    VISION_ENCODER_REGISTRY,
)


@STAGE_REGISTRY.register(
    "build_model",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "model"),
)
class BuildModelStage(ExperimentStage):
    """Build and place the configured model on the resolved device."""

    requires = ("device",)
    provides = ("model",)

    def run(self, context: ExperimentContext) -> None:
        model_cfg = dict(context.config.get("model", {}))
        model_type, kwargs = component_config(
            model_cfg,
            type_keys=("type", "name"),
            default_type="tiny_gpt",
        )
        model = build_model(model_type, **kwargs).to(context.require("device"))
        context.provide("model", model)
        context.track_component("model", MODEL_REGISTRY, str(model_type))

        nested_components = (
            ("block", BLOCK_REGISTRY, model_cfg.get("block_type"), "transformer"),
            ("attention", ATTENTION_REGISTRY, model_cfg.get("attention_type"), "mha"),
            ("norm", NORM_REGISTRY, model_cfg.get("norm_type"), "layernorm"),
            ("mlp", MLP_REGISTRY, model_cfg.get("mlp_type"), "gelu"),
        )
        for role, registry, selector, default in nested_components:
            name = selected_name(selector, default)
            if name in registry:
                context.track_component(role, registry, name)

        selection_provider = getattr(model, "component_selections", None)
        component_selections = (
            selection_provider() if callable(selection_provider) else {}
        )
        multimodal_registries = {
            "vision_encoder": VISION_ENCODER_REGISTRY,
            "multimodal_projector": MULTIMODAL_PROJECTOR_REGISTRY,
            "multimodal_resampler": MULTIMODAL_RESAMPLER_REGISTRY,
            "multimodal_fusion": MULTIMODAL_FUSION_REGISTRY,
            "vision_block": BLOCK_REGISTRY,
            "vision_attention": ATTENTION_REGISTRY,
            "vision_norm": NORM_REGISTRY,
            "vision_mlp": MLP_REGISTRY,
        }
        for role, name in component_selections.items():
            registry = multimodal_registries.get(role)
            if registry is not None and name in registry:
                context.track_component(role, registry, name)
