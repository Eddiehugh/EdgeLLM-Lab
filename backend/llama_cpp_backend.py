"""llama.cpp / GGUF backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend


@BACKEND_REGISTRY.register("llama_cpp")
class LlamaCppBackend(InferenceBackend):
    """Adapter boundary for llama.cpp or llama-cpp-python."""

    def load_model(self, model_path: str):
        raise NotImplementedError("llama.cpp backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("llama.cpp backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("llama.cpp backend benchmark is not wired yet")
