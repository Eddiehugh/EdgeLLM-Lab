"""Sampling strategies for autoregressive generation."""

from __future__ import annotations

import torch

from core.registry import Registry, build_from_config


SAMPLER_REGISTRY = Registry("sampler")


def _select_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits[:, -1, :]
    return logits


@SAMPLER_REGISTRY.register("greedy", capabilities=("deterministic",))
class GreedySampler:
    """Always choose the highest-probability token."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(_select_logits(logits), dim=-1)


@SAMPLER_REGISTRY.register("multinomial", capabilities=("stochastic", "temperature"))
class MultinomialSampler:
    """Sample from the full token distribution."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(temperature, 1e-6)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = _select_logits(logits) / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


@SAMPLER_REGISTRY.register("top_k", capabilities=("stochastic", "top_k", "temperature"))
class TopKSampler:
    """Sample only from the top-k tokens."""

    def __init__(self, k: int = 50, temperature: float = 1.0):
        self.k = k
        self.temperature = max(temperature, 1e-6)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = _select_logits(logits) / self.temperature
        k = min(self.k, logits.size(-1))
        values, indices = torch.topk(logits, k=k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        next_index = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, next_index).squeeze(-1)


@SAMPLER_REGISTRY.register("top_p", capabilities=("stochastic", "top_p", "temperature"))
class TopPSampler:
    """Nucleus sampling."""

    def __init__(self, p: float = 0.95, temperature: float = 1.0):
        self.p = p
        self.temperature = max(temperature, 1e-6)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = _select_logits(logits) / self.temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > self.p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(sorted_logits, dim=-1)
        next_index = torch.multinomial(probs, num_samples=1)
        return sorted_indices.gather(-1, next_index).squeeze(-1)


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Backward-compatible greedy sampling function."""
    return GreedySampler()(logits)


def build_sampler(sampler_type: str | dict = "greedy", **kwargs):
    """Build a sampler by name."""
    return build_from_config(
        SAMPLER_REGISTRY,
        sampler_type,
        default_type="greedy",
        **kwargs,
    )
