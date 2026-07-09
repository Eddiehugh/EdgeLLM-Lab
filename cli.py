"""Command line interface for EdgeLLM-Lab."""

from __future__ import annotations

import argparse
import json
from typing import Any

from core.config import load_config, with_overrides
from experiments import run_experiment
from integrations import build_integration, integration_snapshot


def _load_config_with_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return with_overrides(config, args.override or [])


def _load_all_builtin_components() -> None:
    from backend import build_backend
    from compression import build_quantizer
    from models import build_model

    build_model(
        "tiny_gpt",
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        max_position_embeddings=8,
    )
    build_backend("torch")
    build_quantizer("int8")


def _component_snapshot() -> dict[str, tuple[str, ...]]:
    _load_all_builtin_components()

    from backend import BACKEND_REGISTRY
    from benchmark import BENCHMARK_REGISTRY
    from compression import QUANTIZER_REGISTRY
    from data import DATALOADER_REGISTRY, DATASET_REGISTRY, TOKENIZER_REGISTRY
    from inference import KV_CACHE_REGISTRY, SAMPLER_REGISTRY
    from models import MODEL_REGISTRY
    from modules import (
        ATTENTION_REGISTRY,
        BLOCK_REGISTRY,
        MLP_REGISTRY,
        NORM_REGISTRY,
        POSITION_ENCODING_REGISTRY,
    )
    from training import LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

    return {
        "attention": ATTENTION_REGISTRY.names(),
        "mlp": MLP_REGISTRY.names(),
        "norm": NORM_REGISTRY.names(),
        "block": BLOCK_REGISTRY.names(),
        "position_encoding": POSITION_ENCODING_REGISTRY.names(),
        "model": MODEL_REGISTRY.names(),
        "loss": LOSS_REGISTRY.names(),
        "optimizer": OPTIMIZER_REGISTRY.names(),
        "scheduler": SCHEDULER_REGISTRY.names(),
        "sampler": SAMPLER_REGISTRY.names(),
        "kv_cache": KV_CACHE_REGISTRY.names(),
        "backend": BACKEND_REGISTRY.names(),
        "benchmark": BENCHMARK_REGISTRY.names(),
        "dataset": DATASET_REGISTRY.names(),
        "dataloader": DATALOADER_REGISTRY.names(),
        "tokenizer": TOKENIZER_REGISTRY.names(),
        "quantizer": QUANTIZER_REGISTRY.names(),
    }


def train_command(args: argparse.Namespace) -> int:
    result = run_experiment(_load_config_with_overrides(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def list_components_command(_: argparse.Namespace) -> int:
    for component_type, names in _component_snapshot().items():
        print(f"{component_type}: {', '.join(names)}")
    return 0


def list_integrations_command(args: argparse.Namespace) -> int:
    snapshot = integration_snapshot(external_root=args.external_root)
    for name, info in snapshot.items():
        modes = ", ".join(info["modes"])
        available = "available" if info["available"] else "not linked"
        print(
            f"{name}: {info['purpose']} [{modes}; {available}; "
            f"repo={info['expected_repo_path']}]"
        )
    return 0


def integration_info_command(args: argparse.Namespace) -> int:
    adapter = build_integration(
        args.name,
        local_path=args.local_path,
        external_root=args.external_root,
    )
    info = adapter.validate()
    if args.templates:
        info["config_templates"] = adapter.config_templates()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def smoke_command(args: argparse.Namespace) -> int:
    config = {
        "experiment": {
            "name": "smoke-test",
            "output_dir": args.output_dir,
        },
        "runtime": {
            "device": "cpu",
            "seed": 42,
        },
        "model": {
            "name": "tiny_gpt",
            "vocab_size": 256,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "max_position_embeddings": 16,
            "attention_type": "mha",
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
        },
        "data": {
            "tokenizer_type": "char",
            "text": "EdgeLLM Lab modular smoke test.\n" * 32,
            "block_size": 16,
        },
        "loss": {"type": "causal_lm"},
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-3,
            "max_steps": args.steps,
            "optimizer": {"type": "adamw", "weight_decay": 0.0},
            "scheduler": {"type": "constant"},
        },
    }
    result = run_experiment(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgellm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run an experiment config")
    train.add_argument("-c", "--config", required=True, help="Path to YAML/JSON config")
    train.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Override config values with key=value dot paths",
    )
    train.set_defaults(func=train_command)

    smoke = subparsers.add_parser("smoke", help="Run a tiny CPU smoke experiment")
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--output-dir", default="runs")
    smoke.set_defaults(func=smoke_command)

    list_components = subparsers.add_parser(
        "list-components",
        help="Show registered component names",
    )
    list_components.set_defaults(func=list_components_command)

    list_integrations = subparsers.add_parser(
        "list-integrations",
        help="Show known external project integrations",
    )
    list_integrations.add_argument("--external-root", default="external_projects")
    list_integrations.set_defaults(func=list_integrations_command)

    integration_info = subparsers.add_parser(
        "integration-info",
        help="Show metadata for one external project integration",
    )
    integration_info.add_argument("name", help="Integration name, such as nanogpt")
    integration_info.add_argument("--local-path", default=None)
    integration_info.add_argument("--external-root", default="external_projects")
    integration_info.add_argument("--templates", action="store_true")
    integration_info.set_defaults(func=integration_info_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
