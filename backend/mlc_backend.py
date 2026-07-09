"""MLC LLM backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend


@BACKEND_REGISTRY.register("mlc")
class MLCBackend(InferenceBackend):
    """Adapter boundary for MLC LLM."""

    def load_model(self, model_path: str):
        raise NotImplementedError("MLC backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("MLC backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("MLC backend benchmark is not wired yet")
