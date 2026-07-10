"""llama.cpp / GGUF backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend
from core import Maturity, ProjectLevel


@BACKEND_REGISTRY.register(
    "llama_cpp",
    level=ProjectLevel.WORK,
    maturity=Maturity.PLANNED,
    capabilities=("gguf", "quantized_inference", "edge_runtime"),
    requires=("llama.cpp",),
)
class LlamaCppBackend(InferenceBackend):
    """Adapter boundary for llama.cpp or llama-cpp-python."""

    def load_model(self, model_path: str):
        raise NotImplementedError("llama.cpp backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("llama.cpp backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("llama.cpp backend benchmark is not wired yet")
