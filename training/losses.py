"""Replaceable training losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import Maturity, ProjectLevel
from core.registry import Registry, build_from_config


LOSS_REGISTRY = Registry[nn.Module]("loss")


@LOSS_REGISTRY.register(
    "causal_lm",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("causal_lm", "supervised"),
)
@LOSS_REGISTRY.register(
    "cross_entropy",
    level=ProjectLevel.LEARN,
    maturity=Maturity.VERIFIED,
    capabilities=("supervised",),
)
class CausalLMLoss(nn.Module):
    """Next-token cross entropy for causal language modeling."""

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        shift_labels: bool | None = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.shift_labels = shift_labels

    def _should_shift(self, labels_provided: bool) -> bool:
        if self.shift_labels is not None:
            return self.shift_labels
        return not labels_provided

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        labels_provided = labels is not None
        if labels is None:
            if input_ids is None:
                raise ValueError("CausalLMLoss requires labels or input_ids")
            labels = input_ids

        if self._should_shift(labels_provided):
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


@LOSS_REGISTRY.register(
    "z_loss",
    level=ProjectLevel.LEARN,
    maturity=Maturity.EXPERIMENTAL,
    capabilities=("causal_lm", "logit_regularization"),
)
class ZLoss(CausalLMLoss):
    """Causal LM loss with a logit normalization penalty."""

    def __init__(self, z_loss_weight: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.z_loss_weight = z_loss_weight

    def forward(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        ce_loss = super().forward(logits, **kwargs)
        z_loss = logits.logsumexp(dim=-1).pow(2).mean()
        return ce_loss + self.z_loss_weight * z_loss


@LOSS_REGISTRY.register(
    "distillation",
    level=ProjectLevel.LEARN,
    maturity=Maturity.EXPERIMENTAL,
    capabilities=("causal_lm", "knowledge_distillation"),
    requires=("teacher_logits",),
)
class DistillationLoss(CausalLMLoss):
    """Blend causal LM loss with teacher-student KL divergence."""

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if teacher_logits is None:
            raise ValueError("DistillationLoss requires teacher_logits")

        ce_loss = super().forward(logits, **kwargs)
        labels_provided = kwargs.get("labels") is not None
        if self._should_shift(labels_provided):
            student_logits = logits[:, :-1, :]
            teacher_logits = teacher_logits[:, :-1, :]
        else:
            student_logits = logits
        student = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student, teacher, reduction="batchmean")
        kl_loss = kl_loss * self.temperature * self.temperature
        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss


def build_loss(loss_type: str | dict = "causal_lm", **kwargs) -> nn.Module:
    """Build a loss module by name."""
    return build_from_config(
        LOSS_REGISTRY,
        loss_type,
        default_type="causal_lm",
        **kwargs,
    )
