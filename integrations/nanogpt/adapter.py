"""nanoGPT integration adapter."""

from __future__ import annotations

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import INTEGRATION_REGISTRY


@INTEGRATION_REGISTRY.register("nanogpt")
class NanoGPTIntegration(IntegrationAdapter):
    """Use nanoGPT as the fastest minimal train/generate reference."""

    info = IntegrationInfo(
        name="nanogpt",
        purpose="Fast reference for the minimal pretrain/generate loop.",
        project_url="https://github.com/karpathy/nanoGPT",
        modes=("reference", "adapter", "absorb"),
        capabilities=(
            "minimal_training_loop_reference",
            "gpt_config_mapping",
            "checkpoint_conversion_target",
        ),
        recommended_first_step=(
            "Use it as a reference implementation, then port the loop into "
            "experiments/runner.py instead of depending on its scripts directly."
        ),
        notes=(
            "Best for learning the closed loop quickly.",
            "Do not make nanoGPT the long-term core framework.",
        ),
    )

    def config_templates(self):
        return {
            "tiny_gpt_like": {
                "model": {
                    "name": "tiny_gpt",
                    "attention_type": "mha",
                    "norm_type": "layernorm",
                    "mlp_type": "gelu",
                },
                "loss": {"type": "causal_lm"},
            }
        }
