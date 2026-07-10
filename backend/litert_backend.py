"""LiteRT backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend
from core import Maturity, ProjectLevel


@BACKEND_REGISTRY.register(
    "litert",
    level=ProjectLevel.WORK,
    maturity=Maturity.PLANNED,
    capabilities=("edge_runtime", "mobile"),
    requires=("litert",),
)
class LiteRTBackend(InferenceBackend):
    """Adapter boundary for LiteRT."""

    def load_model(self, model_path: str):
        raise NotImplementedError("LiteRT backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("LiteRT backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("LiteRT backend benchmark is not wired yet")
