"""Readable AdamW reference implementation for learning and parity tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from core import Maturity, ProjectLevel
from training.optimizers.registry import OPTIMIZER_REGISTRY


class ReferenceAdamW(torch.optim.Optimizer):
    """Minimal decoupled AdamW implementation without fused backend paths."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, object]],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = {
            "lr": lr,
            "betas": tuple(betas),
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Perform one AdamW update using the equations directly."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            learning_rate = group["lr"]
            weight_decay = group["weight_decay"]
            epsilon = group["eps"]
            maximize = group["maximize"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("ReferenceAdamW does not support sparse gradients")
                if maximize:
                    gradient = -gradient

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                parameter.mul_(1.0 - learning_rate * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = learning_rate / bias_correction1
                denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5)
                denominator.add_(epsilon)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss


@OPTIMIZER_REGISTRY.register(
    "reference_adamw",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("gradient_optimization", "adamw", "reference_implementation"),
)
def build_reference_adamw(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, object]],
    **kwargs: object,
) -> ReferenceAdamW:
    """Build the readable Level 1 AdamW implementation."""

    return ReferenceAdamW(params, **kwargs)
