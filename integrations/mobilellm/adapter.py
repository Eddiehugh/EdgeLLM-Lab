"""MobileLLM integration adapter."""

from __future__ import annotations

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import INTEGRATION_REGISTRY


@INTEGRATION_REGISTRY.register("mobilellm")
class MobileLLMIntegration(IntegrationAdapter):
    """Use MobileLLM as an edge-oriented architecture design reference."""

    info = IntegrationInfo(
        name="mobilellm",
        purpose="Reference for sub-billion-parameter on-device architecture choices.",
        project_url="https://arxiv.org/abs/2402.14905",
        modes=("absorb", "benchmark"),
        capabilities=(
            "edge_architecture_reference",
            "deep_thin_model_design",
            "mobile_latency_benchmark_target",
        ),
        recommended_first_step=(
            "Absorb architectural ideas into local modules, then compare latency, "
            "memory, and quality under the same benchmark suite."
        ),
        notes=(
            "Most useful as a design reference rather than a direct dependency.",
            "Keep mobile-specific deployment code under backend/ or deploy/.",
        ),
    )
