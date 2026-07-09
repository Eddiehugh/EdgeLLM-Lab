"""ONNX Runtime backend adapter placeholder."""

from backend.base import BACKEND_REGISTRY, InferenceBackend


@BACKEND_REGISTRY.register("onnx")
class ONNXBackend(InferenceBackend):
    """Adapter boundary for ONNX Runtime inference."""

    def load_model(self, model_path: str):
        raise NotImplementedError("ONNX backend integration is not wired yet")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("ONNX backend integration is not wired yet")

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("ONNX backend benchmark is not wired yet")
