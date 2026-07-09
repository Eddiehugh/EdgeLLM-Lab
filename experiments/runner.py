"""End-to-end experiment runner."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from core import (
    component_config,
    count_parameters,
    load_extensions_from_config,
    model_size_bytes,
    resolve_device,
    set_seed,
    timed,
)
from data import build_dataloader, build_dataset, build_tokenizer
from experiments.report import build_report
from experiments.run_store import RunStore
from models import build_model
from training import build_loss, build_optimizer, build_scheduler


class ExperimentRunner:
    """Build components from config, run training, and persist artifacts."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        load_extensions_from_config(self.config)

    def run(self) -> dict[str, Any]:
        experiment_cfg = dict(self.config.get("experiment", {}))
        name = experiment_cfg.get("name", self.config.get("name", "experiment"))
        output_root = experiment_cfg.get("output_dir", self.config.get("output_dir", "runs"))

        run = RunStore(output_root).create_run(str(name))
        run.write_config(self.config)

        set_seed(self._runtime_cfg().get("seed", self.config.get("seed", 42)))
        device = resolve_device(self._runtime_cfg().get("device", "auto"))

        tokenizer = self._build_tokenizer()
        dataset = self._build_dataset(tokenizer)
        dataloader = self._build_dataloader(dataset)
        model = self._build_model().to(device)
        loss_fn = self._build_loss()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        metrics = self._train(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            device=device,
        )
        metrics.update(
            {
                "run_id": run.run_id,
                "device": str(device),
                "parameter_count": count_parameters(model),
                "trainable_parameter_count": count_parameters(model, trainable_only=True),
                "model_size_mb": model_size_bytes(model) / (1024 * 1024),
            }
        )

        if bool(self._training_cfg().get("save_checkpoint", False)):
            checkpoint_path = run.write_checkpoint(model)
            metrics["checkpoint_path"] = str(checkpoint_path)

        run.write_metrics(metrics)
        report = build_report(title=str(name), config=self.config, metrics=metrics)
        report_path = run.write_text("report.md", report)
        metrics["report_path"] = str(report_path)
        return {"run_dir": str(run.path), "metrics": metrics}

    def _runtime_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("runtime", {}))

    def _training_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("training", {}))

    def _data_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("data", {}))

    def _build_tokenizer(self):
        data_cfg = self._data_cfg()
        tokenizer_cfg = data_cfg.get("tokenizer", data_cfg.get("tokenizer_type", "char"))
        return build_tokenizer(tokenizer_cfg)

    def _load_text(self) -> str:
        data_cfg = self._data_cfg()
        if "text" in data_cfg:
            return str(data_cfg["text"])
        if "text_file" in data_cfg:
            return Path(data_cfg["text_file"]).read_text(encoding="utf-8")
        return "EdgeLLM Lab default tiny text corpus.\n" * 64

    def _build_dataset(self, tokenizer):
        data_cfg = self._data_cfg()
        block_size = int(
            data_cfg.get(
                "block_size",
                self.config.get("model", {}).get("max_position_embeddings", 128),
            )
        )
        token_ids = data_cfg.get("token_ids")
        if token_ids is None:
            token_ids = tokenizer.encode(self._load_text())

        dataset_cfg = data_cfg.get("dataset", data_cfg.get("dataset_type", "causal_lm"))
        return build_dataset(dataset_cfg, token_ids=token_ids, block_size=block_size)

    def _build_dataloader(self, dataset):
        training_cfg = self._training_cfg()
        data_cfg = self._data_cfg()
        dataloader_cfg = data_cfg.get("dataloader", data_cfg.get("dataloader_type", "torch"))
        batch_size = int(training_cfg.get("batch_size", data_cfg.get("batch_size", 1)))
        return build_dataloader(
            dataloader_cfg,
            dataset=dataset,
            batch_size=batch_size,
            shuffle=bool(data_cfg.get("shuffle", True)),
        )

    def _build_model(self) -> torch.nn.Module:
        model_cfg = dict(self.config.get("model", {}))
        model_type, kwargs = component_config(
            model_cfg,
            type_keys=("type", "name"),
            default_type="tiny_gpt",
        )
        return build_model(model_type, **kwargs)

    def _build_loss(self):
        loss_cfg = self.config.get("loss", "causal_lm")
        return build_loss(loss_cfg)

    def _build_optimizer(self, model: torch.nn.Module):
        training_cfg = self._training_cfg()
        optimizer_cfg = training_cfg.get("optimizer", {"type": "adamw"})
        if isinstance(optimizer_cfg, dict) and "lr" not in optimizer_cfg:
            optimizer_cfg = dict(optimizer_cfg)
            optimizer_cfg["lr"] = training_cfg.get("learning_rate", 3e-4)
        return build_optimizer(optimizer_cfg, params=model.parameters())

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        training_cfg = self._training_cfg()
        scheduler_cfg = training_cfg.get("scheduler", {"type": "constant"})
        if isinstance(scheduler_cfg, dict) and "max_steps" not in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            scheduler_cfg["max_steps"] = training_cfg.get("max_steps", 1)
        return build_scheduler(scheduler_cfg, optimizer=optimizer)

    def _train(
        self,
        *,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloader,
        device: torch.device,
    ) -> dict[str, Any]:
        training_cfg = self._training_cfg()
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

        final_loss = losses[-1] if losses else float("nan")
        return {
            "steps": step,
            "tokens_seen": tokens_seen,
            "final_train_loss": final_loss,
            "logged_train_losses": losses,
            "runtime_seconds": timer.elapsed_seconds,
            "tokens_per_second": tokens_seen / max(timer.elapsed_seconds, 1e-12),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one experiment from a config mapping."""

    return ExperimentRunner(config).run()
