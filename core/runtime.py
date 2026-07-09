"""Runtime utilities shared by training, inference, and benchmark code."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def model_size_bytes(model: torch.nn.Module) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


@dataclass
class Timer:
    start_time: float
    end_time: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


@contextmanager
def timed() -> Iterator[Timer]:
    timer = Timer(start_time=time.perf_counter())
    try:
        yield timer
    finally:
        timer.end_time = time.perf_counter()
