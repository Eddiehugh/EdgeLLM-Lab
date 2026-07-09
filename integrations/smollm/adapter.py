"""SmolLM integration adapter."""

from __future__ import annotations

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import INTEGRATION_REGISTRY


@INTEGRATION_REGISTRY.register("smollm")
class SmolLMIntegration(IntegrationAdapter):
    """Use SmolLM as a small-model-family experiment reference."""

    info = IntegrationInfo(
        name="smollm",
        purpose="Reference for comparing small model family scales and recipes.",
        project_url="https://huggingface.co/HuggingFaceTB",
        modes=("adapter", "benchmark", "absorb"),
        capabilities=(
            "small_model_family_configs",
            "hf_checkpoint_reference",
            "benchmark_comparison_target",
        ),
        recommended_first_step=(
            "Create local configs that match the target SmolLM scale, then run "
            "the same benchmark metrics against local and external checkpoints."
        ),
        notes=(
            "Useful for model-size and data-recipe comparisons.",
            "Treat HF checkpoints as inputs to adapters, not as core code.",
        ),
    )
