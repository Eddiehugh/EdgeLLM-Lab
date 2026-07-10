"""PyTorch inference backend."""

from __future__ import annotations

import torch

from backend.base import BACKEND_REGISTRY, InferenceBackend
from core import Maturity, ProjectLevel
from inference.sampler import build_sampler


@BACKEND_REGISTRY.register(
    "torch",
    level=ProjectLevel.WORK,
    maturity=Maturity.EXPERIMENTAL,
    capabilities=("eager", "generate", "pytorch"),
    requires=("torch",),
)
class TorchBackend(InferenceBackend):
    """Minimal backend for in-process PyTorch models."""

    def __init__(self, model=None, tokenizer=None, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def load_model(self, model_path: str):
        raise NotImplementedError("TorchBackend checkpoint loading is not wired yet")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        sampler: str | dict = "greedy",
        **sampler_kwargs,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("TorchBackend requires both model and tokenizer")

        sample = build_sampler(sampler, **sampler_kwargs)
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
            device=self.device,
        )
        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            next_token = sample(logits).view(1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(input_ids[0].tolist())

    def benchmark(self, prompts: list[str]):
        raise NotImplementedError("TorchBackend benchmark will be added in benchmark/")
