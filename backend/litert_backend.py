"""LiteRT backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend


@BACKEND_REGISTRY.register("litert")
class LiteRTBackend(InferenceBackend):
    """Adapter boundary for LiteRT."""

    def load_model(self, model_path: str):
        raise NotImplementedError("LiteRT backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("LiteRT backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("LiteRT backend benchmark is not wired yet")
