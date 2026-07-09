"""llama.cpp integration adapter."""

from __future__ import annotations

from integrations.base import IntegrationAdapter, IntegrationInfo
from integrations.registry import INTEGRATION_REGISTRY


@INTEGRATION_REGISTRY.register("llama_cpp", "llama.cpp")
class LlamaCppIntegration(IntegrationAdapter):
    """Use llama.cpp as the production-grade edge inference runtime reference."""

    info = IntegrationInfo(
        name="llama_cpp",
        purpose="Reference backend for GGUF export, quantized inference, and edge benchmarking.",
        project_url="https://github.com/ggml-org/llama.cpp",
        modes=("backend", "export", "benchmark"),
        capabilities=(
            "gguf_runtime_target",
            "quantized_inference_backend",
            "edge_latency_benchmark_target",
        ),
        recommended_first_step=(
            "Keep model training local, export or convert weights to a llama.cpp "
            "compatible format, then compare runtime metrics through backend/."
        ),
        notes=(
            "Use backend/llama_cpp_backend.py as the runtime boundary.",
            "Do not let llama.cpp-specific formats leak into model definitions.",
        ),
    )
