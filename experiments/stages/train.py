"""Default autoregressive training stage."""

from __future__ import annotations

import itertools

from core import timed
from core.specs import Maturity, ProjectLevel
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage


@STAGE_REGISTRY.register(
    "train",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "training_loop"),
)
class TrainStage(ExperimentStage):
    """Run the default autoregressive optimization loop."""

    requires = ("model", "loss", "optimizer", "scheduler", "dataloader", "device")

    def run(self, context: ExperimentContext) -> None:
        model = context.require("model")
        loss_fn = context.require("loss")
        optimizer = context.require("optimizer")
        scheduler = context.require("scheduler")
        dataloader = context.require("dataloader")
        device = context.require("device")
        training_cfg = dict(context.config.get("training", {}))

        max_steps = int(training_cfg.get("max_steps", 1))
        grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
        log_interval = int(training_cfg.get("log_interval", max(1, max_steps)))
        if len(dataloader) == 0:
            raise ValueError("Dataloader is empty. Increase data length or reduce block_size.")

        model.train()
        losses: list[float] = []
        tokens_seen = 0
        step = 0
        batch_iter = itertools.cycle(dataloader)

        with timed() as timer:
            while step < max_steps:
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                for _ in range(grad_accum_steps):
                    batch = next(batch_iter)
                    input_ids = batch["input_ids"].to(device)
                    labels = batch.get("labels")
                    labels = labels.to(device) if labels is not None else None
                    logits = model(input_ids)
                    loss = loss_fn(logits, labels=labels, input_ids=input_ids)
                    (loss / grad_accum_steps).backward()
                    accum_loss += float(loss.detach())
                    tokens_seen += int(input_ids.numel())

                optimizer.step()
                scheduler.step()
                step += 1
                loss_value = accum_loss / grad_accum_steps
                if step == 1 or step == max_steps or step % log_interval == 0:
                    losses.append(loss_value)

        context.metrics.update(
            {
                "steps": step,
                "tokens_seen": tokens_seen,
                "final_train_loss": losses[-1] if losses else float("nan"),
                "logged_train_losses": losses,
                "runtime_seconds": timer.elapsed_seconds,
                "tokens_per_second": tokens_seen / max(timer.elapsed_seconds, 1e-12),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
