"""ExecuTorch backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend
from core import Maturity, ProjectLevel


@BACKEND_REGISTRY.register(
    "executorch",
    level=ProjectLevel.WORK,
    maturity=Maturity.PLANNED,
    capabilities=("pte", "edge_runtime"),
    requires=("executorch",),
)
class ExecuTorchBackend(InferenceBackend):
    """Adapter boundary for ExecuTorch."""

    def load_model(self, model_path: str):
        raise NotImplementedError("ExecuTorch backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("ExecuTorch backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("ExecuTorch backend benchmark is not wired yet")
