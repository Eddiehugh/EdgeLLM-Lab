"""Build tokenizer, dataset, and dataloader components."""

from __future__ import annotations

from pathlib import Path

from core.specs import Maturity, ProjectLevel
from data import (
    DATALOADER_REGISTRY,
    DATASET_REGISTRY,
    TOKENIZER_REGISTRY,
    build_dataloader,
    build_dataset,
    build_tokenizer,
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

        block_size = int(
            data_cfg.get("block_size", model_cfg.get("max_position_embeddings", 128))
        )
        token_ids = data_cfg.get("token_ids")
        if token_ids is None:
            token_ids = tokenizer.encode(self._load_text(data_cfg))

        dataset_cfg = data_cfg.get("dataset", data_cfg.get("dataset_type", "causal_lm"))
        dataset = context.provide(
            "dataset",
            build_dataset(dataset_cfg, token_ids=token_ids, block_size=block_size),
        )
        context.track_component(
            "dataset",
            DATASET_REGISTRY,
            selected_name(dataset_cfg, "causal_lm"),
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
