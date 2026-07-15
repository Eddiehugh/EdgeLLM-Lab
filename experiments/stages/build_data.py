"""Build tokenizer, dataset, and dataloader components."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from core.specs import Maturity, ProjectLevel
from data import (
    DATALOADER_REGISTRY,
    DATASET_REGISTRY,
    TOKENIZER_REGISTRY,
    build_dataloader,
    build_dataset,
    build_tokenizer,
    load_builtin_datasets,
)
from experiments.context import ExperimentContext
from experiments.stage import STAGE_REGISTRY, ExperimentStage
from experiments.stages.common import selected_name


@STAGE_REGISTRY.register(
    "build_data",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.VERIFIED,
    capabilities=("pipeline", "data"),
)
class BuildDataStage(ExperimentStage):
    """Build the tokenizer, dataset, and dataloader."""

    provides = ("tokenizer", "dataset", "dataloader")

    @staticmethod
    def _load_text(data_cfg: dict) -> str:
        if "text" in data_cfg:
            return str(data_cfg["text"])
        if "text_file" in data_cfg:
            return Path(data_cfg["text_file"]).read_text(encoding="utf-8")
        return "EdgeLLM Lab default tiny text corpus.\n" * 64

    def run(self, context: ExperimentContext) -> None:
        data_cfg = dict(context.config.get("data", {}))
        training_cfg = dict(context.config.get("training", {}))
        model_cfg = dict(context.config.get("model", {}))

        tokenizer_cfg = data_cfg.get("tokenizer", data_cfg.get("tokenizer_type", "char"))
        tokenizer = context.provide("tokenizer", build_tokenizer(tokenizer_cfg))
        context.track_component(
            "tokenizer",
            TOKENIZER_REGISTRY,
            selected_name(tokenizer_cfg, "char"),
        )

        dataset_cfg = data_cfg.get("dataset", data_cfg.get("dataset_type", "causal_lm"))
        load_builtin_datasets()
        dataset_name = selected_name(dataset_cfg, "causal_lm")
        dataset_spec = DATASET_REGISTRY.get_spec(dataset_name)
        configured_dataset_values = (
            dict(dataset_cfg) if isinstance(dataset_cfg, Mapping) else {}
        )
        configured_dataset_values.pop("type", None)
        configured_dataset_values.pop("name", None)

        block_size = int(
            data_cfg.get("block_size", model_cfg.get("max_position_embeddings", 128))
        )
        def token_ids() -> Any:
            configured = data_cfg.get("token_ids")
            if configured is not None:
                return configured
            return tokenizer.encode(self._load_text(data_cfg))

        providers: dict[str, Callable[[], Any]] = {
            "block_size": lambda: block_size,
            "token_ids": token_ids,
            "tokenizer": lambda: tokenizer,
            "vocab_size": lambda: int(model_cfg["vocab_size"]),
        }
        dataset_kwargs: dict[str, Any] = {}
        for requirement in dataset_spec.requires:
            if requirement in configured_dataset_values:
                continue
            try:
                dataset_kwargs[requirement] = providers[requirement]()
            except KeyError as exc:
                raise ValueError(
                    f"Dataset '{dataset_name}' requires '{requirement}', but it is "
                    "not present in data.dataset and has no pipeline provider"
                ) from exc

        dataset = context.provide(
            "dataset",
            build_dataset(dataset_cfg, **dataset_kwargs),
        )
        context.track_component(
            "dataset",
            DATASET_REGISTRY,
            dataset_name,
        )

        dataloader_cfg = data_cfg.get(
            "dataloader", data_cfg.get("dataloader_type", "torch")
        )
        dataloader = build_dataloader(
            dataloader_cfg,
            dataset=dataset,
            batch_size=int(training_cfg.get("batch_size", data_cfg.get("batch_size", 1))),
            shuffle=bool(data_cfg.get("shuffle", True)),
        )
        context.provide("dataloader", dataloader)
        context.track_component(
            "dataloader",
            DATALOADER_REGISTRY,
            selected_name(dataloader_cfg, "torch"),
        )
