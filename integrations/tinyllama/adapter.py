"""TinyLlama integration adapter."""

from __future__ import annotations

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import INTEGRATION_REGISTRY


@INTEGRATION_REGISTRY.register("tinyllama")
class TinyLlamaIntegration(IntegrationAdapter):
    """Use TinyLlama as the LLaMA-like architecture reference."""

    info = IntegrationInfo(
        name="tinyllama",
        purpose="Reference for compact LLaMA-style model structure.",
        project_url="https://github.com/jzhang38/TinyLlama",
        modes=("adapter", "absorb", "benchmark"),
        capabilities=(
            "llama_like_config_mapping",
            "rmsnorm_rope_swiglu_gqa_reference",
            "checkpoint_conversion_target",
        ),
        recommended_first_step=(
            "Map TinyLlama architecture fields into local llama_like configs, "
            "then implement missing modules locally."
        ),
        notes=(
            "Use it to validate LLaMA-like structure choices.",
            "Keep checkpoint conversion separate from model implementation.",
        ),
    )

    def config_templates(self):
        return {
            "llama_like_small": {
                "model": {
                    "name": "tiny_gpt",
                    "attention_type": "mha",
                    "norm_type": "rmsnorm",
                    "mlp_type": "swiglu",
                    "tie_word_embeddings": True,
                }
            }
        }
