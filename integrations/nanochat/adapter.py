"""nanochat integration adapter."""

from __future__ import annotations

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import INTEGRATION_REGISTRY


@INTEGRATION_REGISTRY.register("nanochat")
class NanochatIntegration(IntegrationAdapter):
    """Run nanochat independently as the first external cloud workload."""

    info = IntegrationInfo(
        name="nanochat",
        purpose="Reference full-stack pretraining, evaluation, SFT, RL, and chat loop.",
        project_url="https://github.com/karpathy/nanochat",
        modes=("wrap", "benchmark", "absorb"),
        capabilities=(
            "external_cloud_training",
            "full_stack_llm_reference",
            "checkpoint_and_recipe_comparison",
        ),
        recommended_first_step=(
            "Run configs/execution/autodl_nanochat_smoke.yaml at a pinned revision, "
            "then fork nanochat and repeat after one local model change."
        ),
        local_path="external_projects/nanochat",
        notes=(
            "nanochat remains a standalone Git checkout and dependency environment.",
            "EdgeLLM-Lab records the revision, commands, logs, and selected artifacts.",
        ),
    )

    def config_templates(self):
        return {
            "autodl_smoke": {
                "execution": {
                    "workload": {
                        "type": "external_project",
                        "integration": "nanochat",
                        "source": {"local_path": "external_projects/nanochat"},
                    }
                }
            }
        }
