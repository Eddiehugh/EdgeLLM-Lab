"""Command line interface for EdgeLLM-Lab."""

from __future__ import annotations

import argparse
import json
from typing import Any

from core.config import load_config, with_overrides
from experiments import normalize_experiment_config, run_experiment
from integrations import build_integration, integration_snapshot


def _load_config_with_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return with_overrides(config, args.override or [])


def _load_all_builtin_components() -> None:
    from backend import load_builtin_backends
    from compression import load_builtin_quantizers
    from experiments.stage import load_builtin_stages
    from models import load_builtin_models

    load_builtin_models()
    load_builtin_backends()
    load_builtin_quantizers()
    load_builtin_stages()


def _component_registries():
    _load_all_builtin_components()

    from backend import BACKEND_REGISTRY
    from benchmark import BENCHMARK_REGISTRY
    from compression import QUANTIZER_REGISTRY
    from data import DATALOADER_REGISTRY, DATASET_REGISTRY, TOKENIZER_REGISTRY
    from experiments import STAGE_REGISTRY
    from inference import KV_CACHE_REGISTRY, SAMPLER_REGISTRY
    from models import MODEL_REGISTRY
    from modules import (
        ATTENTION_REGISTRY,
        BLOCK_REGISTRY,
        MLP_REGISTRY,
        MOE_REGISTRY,
        NORM_REGISTRY,
        POSITION_ENCODING_REGISTRY,
    )
    from training import LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

    return {
        "attention": ATTENTION_REGISTRY,
        "mlp": MLP_REGISTRY,
        "moe": MOE_REGISTRY,
        "norm": NORM_REGISTRY,
        "block": BLOCK_REGISTRY,
        "position_encoding": POSITION_ENCODING_REGISTRY,
        "model": MODEL_REGISTRY,
        "loss": LOSS_REGISTRY,
        "optimizer": OPTIMIZER_REGISTRY,
        "scheduler": SCHEDULER_REGISTRY,
        "sampler": SAMPLER_REGISTRY,
        "kv_cache": KV_CACHE_REGISTRY,
        "backend": BACKEND_REGISTRY,
        "benchmark": BENCHMARK_REGISTRY,
        "dataset": DATASET_REGISTRY,
        "dataloader": DATALOADER_REGISTRY,
        "tokenizer": TOKENIZER_REGISTRY,
        "quantizer": QUANTIZER_REGISTRY,
        "experiment_stage": STAGE_REGISTRY,
    }


def _component_snapshot() -> dict[str, tuple[str, ...]]:
    return {
        component_type: registry.canonical_names()
        for component_type, registry in _component_registries().items()
    }


def train_command(args: argparse.Namespace) -> int:
    result = run_experiment(_load_config_with_overrides(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def list_components_command(args: argparse.Namespace) -> int:
    registries = _component_registries()
    if args.as_json:
        snapshot = {
            component_type: registry.snapshot()
            for component_type, registry in registries.items()
        }
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return 0

    if args.details:
        for component_type, registry in registries.items():
            for name, info in registry.snapshot().items():
                capabilities = ",".join(info["capabilities"]) or "-"
                print(
                    f"{component_type}.{name}: level={info['level']} "
                    f"maturity={info['maturity']} capabilities={capabilities}"
                )
        return 0

    for component_type, names in _component_snapshot().items():
        print(f"{component_type}: {', '.join(names)}")
    return 0


def component_info_command(args: argparse.Namespace) -> int:
    registries = _component_registries()
    try:
        registry = registries[args.component_type]
    except KeyError as exc:
        available = ", ".join(registries)
        raise ValueError(
            f"Unknown component type '{args.component_type}'. Available: {available}"
        ) from exc
    print(json.dumps(registry.describe(args.name), indent=2, ensure_ascii=False))
    return 0


def validate_config_command(args: argparse.Namespace) -> int:
    config = normalize_experiment_config(_load_config_with_overrides(args))
    if args.resolved:
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        stages = ", ".join(
            stage
            if isinstance(stage, str)
            else str(stage.get("type", stage.get("name")))
            for stage in config["pipeline"]["stages"]
        )
        print(f"valid schema_version={config['schema_version']} stages={stages}")
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
    list_components.add_argument("--details", action="store_true")
    list_components.add_argument("--json", action="store_true", dest="as_json")
    list_components.set_defaults(func=list_components_command)

    component_info = subparsers.add_parser(
        "component-info",
        help="Show metadata for one registered component",
    )
    component_info.add_argument("component_type")
    component_info.add_argument("name")
    component_info.set_defaults(func=component_info_command)

    validate_config = subparsers.add_parser(
        "validate-config",
        help="Validate and resolve an experiment config without running it",
    )
    validate_config.add_argument("-c", "--config", required=True)
    validate_config.add_argument("-o", "--override", action="append", default=[])
    validate_config.add_argument("--resolved", action="store_true")
    validate_config.set_defaults(func=validate_config_command)

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
