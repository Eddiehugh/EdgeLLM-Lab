"""Command line interface for EdgeLLM-Lab."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
from typing import Any

from core.config import load_config, with_overrides
from execution import (
    ConnectionProfileStore,
    JobState,
    RunManager,
    parse_ssh_command,
    redact_connection,
)
from execution.executors.ssh import SSHExecutor
from execution.metadata import JsonMetadataStore
from experiments import normalize_experiment_config, run_experiment
from integrations import build_integration, integration_snapshot
from reproduction import PaperStudyManager, PaperWorkspace


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
    from training import (
        LOSS_REGISTRY,
        OPTIMIZER_REGISTRY,
        PARAM_GROUP_POLICY_REGISTRY,
        SCHEDULER_REGISTRY,
    )

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
        "optimizer_param_group_policy": PARAM_GROUP_POLICY_REGISTRY,
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


def _run_manager(args: argparse.Namespace) -> RunManager:
    metadata_root = getattr(args, "metadata_root", None)
    if metadata_root:
        return RunManager(metadata_store=JsonMetadataStore(metadata_root))
    return RunManager()


def submit_command(args: argparse.Namespace) -> int:
    manager = _run_manager(args)
    record = manager.submit(_load_config_with_overrides(args))
    if args.wait:
        record = manager.wait(
            record.job_id,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    if args.wait and record.state == JobState.FAILED:
        return 1
    return 0


def job_status_command(args: argparse.Namespace) -> int:
    record = _run_manager(args).status(args.job_id)
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    return 0


def job_logs_command(args: argparse.Namespace) -> int:
    print(_run_manager(args).logs(args.job_id, tail=args.tail))
    return 0


def cancel_job_command(args: argparse.Namespace) -> int:
    record = _run_manager(args).cancel(args.job_id)
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    return 0


def fetch_job_command(args: argparse.Namespace) -> int:
    destination = args.output or str(Path("downloads") / args.job_id)
    path = _run_manager(args).fetch(args.job_id, destination)
    print(str(path))
    return 0


def list_jobs_command(args: argparse.Namespace) -> int:
    records = _run_manager(args).list()
    if args.as_json:
        print(
            json.dumps(
                [record.to_dict() for record in records],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    for record in records:
        print(
            f"{record.job_id}  {record.state.value:<10}  "
            f"{record.executor_type:<18}  {record.name}"
        )
    return 0


def _connection_store(args: argparse.Namespace) -> ConnectionProfileStore:
    return ConnectionProfileStore(
        args.store or Path.cwd() / ".edgellm" / "connections.json"
    )


def connection_set_command(args: argparse.Namespace) -> int:
    store = _connection_store(args)
    values: dict[str, Any] = {}
    if args.ssh_command:
        values.update(parse_ssh_command(args.ssh_command))
    for key in ("host", "user", "port", "identity_file"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.password:
        password = getpass.getpass("SSH password: ")
        if not password:
            raise ValueError("SSH password must not be empty")
        values["password"] = password
    if args.clear_password:
        values["password"] = None
    if args.clear_identity_file:
        values["identity_file"] = None
    if args.accept_new_host_key:
        try:
            existing_options = list(store.get(args.name).get("ssh_options", []))
        except KeyError:
            existing_options = []
        options = list(values.get("ssh_options", existing_options))
        setting = ["-o", "StrictHostKeyChecking=accept-new"]
        if setting[1] not in options:
            options.extend(setting)
        values["ssh_options"] = options
    if not values:
        raise ValueError("Provide --ssh-command or at least one connection field")
    profile = store.set(args.name, values)
    print(
        json.dumps(
            {
                "name": args.name,
                "store": str(store.path),
                "connection": redact_connection(profile),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def connection_show_command(args: argparse.Namespace) -> int:
    store = _connection_store(args)
    print(
        json.dumps(
            {
                "name": args.name,
                "store": str(store.path),
                "connection": redact_connection(store.get(args.name)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def connection_list_command(args: argparse.Namespace) -> int:
    store = _connection_store(args)
    profiles = store.list()
    if args.as_json:
        print(
            json.dumps(
                {
                    name: redact_connection(profile)
                    for name, profile in profiles.items()
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if not profiles:
        print(f"No connection profiles in {store.path}")
        return 0
    for name, profile in profiles.items():
        user = f"{profile.get('user')}@" if profile.get("user") else ""
        port = f":{profile.get('port')}" if profile.get("port") else ""
        print(f"{name}: {user}{profile['host']}{port}")
    return 0


def connection_test_command(args: argparse.Namespace) -> int:
    store = _connection_store(args)
    profile = store.get(args.name)
    profile["command_timeout"] = args.timeout
    result = SSHExecutor(profile).probe()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def connection_remove_command(args: argparse.Namespace) -> int:
    store = _connection_store(args)
    store.remove(args.name)
    print(f"Removed connection profile '{args.name}' from {store.path}")
    return 0


def _paper_workspace(args: argparse.Namespace) -> PaperWorkspace:
    return PaperWorkspace(root=args.paper_root)


def _paper_study_manager(args: argparse.Namespace) -> PaperStudyManager:
    workspace = _paper_workspace(args)
    metadata_root = getattr(args, "metadata_root", None)
    run_manager = None
    if metadata_root:
        run_manager = RunManager(
            metadata_store=JsonMetadataStore(metadata_root),
            project_root=workspace.project_root,
        )
    return PaperStudyManager(
        workspace=workspace,
        run_manager=run_manager,
        state_root=getattr(args, "study_root", None),
    )


def paper_init_command(args: argparse.Namespace) -> int:
    path = _paper_workspace(args).scaffold(
        args.paper_id,
        args.title,
        url=args.url,
        authors=args.author or [],
        year=args.year,
    )
    print(str(path))
    return 0


def paper_list_command(args: argparse.Namespace) -> int:
    papers = _paper_workspace(args).list()
    if args.as_json:
        print(
            json.dumps(
                [
                    {
                        "id": paper.id,
                        "title": paper.title,
                        "year": paper.year,
                        "url": paper.url,
                        "suites": sorted(paper.suites),
                    }
                    for paper in papers
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    for paper in papers:
        print(f"{paper.id:<28} {paper.year or '-':<6} {paper.title}")
    return 0


def paper_info_command(args: argparse.Namespace) -> int:
    paper = _paper_workspace(args).load(args.paper_id)
    print(json.dumps(paper.manifest, indent=2, ensure_ascii=False))
    return 0


def paper_validate_command(args: argparse.Namespace) -> int:
    result = _paper_workspace(args).validate(args.paper_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def paper_study_command(args: argparse.Namespace) -> int:
    study = _paper_study_manager(args).run_study(
        args.paper_id,
        suite_name=args.suite,
        executor=args.executor,
        overrides=args.override or [],
        detach=args.detach,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    print(json.dumps(study, indent=2, ensure_ascii=False))
    return 1 if study["status"] == "failed" else 0


def paper_assess_command(args: argparse.Namespace) -> int:
    study = _paper_study_manager(args).assess_study(
        args.paper_id,
        args.study_id,
        wait=args.wait,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    print(json.dumps(study, indent=2, ensure_ascii=False))
    return 1 if study["status"] == "failed" else 0


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

    submit = subparsers.add_parser(
        "submit",
        help="Submit an experiment through a configured execution backend",
    )
    submit.add_argument("-c", "--config", required=True)
    submit.add_argument("-o", "--override", action="append", default=[])
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("--poll-interval", type=float, default=1.0)
    submit.add_argument("--timeout", type=float, default=None)
    submit.add_argument("--metadata-root", default=None)
    submit.set_defaults(func=submit_command)

    job_status = subparsers.add_parser("status", help="Query one submitted job")
    job_status.add_argument("job_id")
    job_status.add_argument("--metadata-root", default=None)
    job_status.set_defaults(func=job_status_command)

    job_logs = subparsers.add_parser("logs", help="Read provider job logs")
    job_logs.add_argument("job_id")
    job_logs.add_argument("--tail", type=int, default=200)
    job_logs.add_argument("--metadata-root", default=None)
    job_logs.set_defaults(func=job_logs_command)

    cancel_job = subparsers.add_parser("cancel", help="Cancel a submitted job")
    cancel_job.add_argument("job_id")
    cancel_job.add_argument("--metadata-root", default=None)
    cancel_job.set_defaults(func=cancel_job_command)

    fetch_job = subparsers.add_parser("fetch", help="Fetch completed job artifacts")
    fetch_job.add_argument("job_id")
    fetch_job.add_argument("--output", default=None)
    fetch_job.add_argument("--metadata-root", default=None)
    fetch_job.set_defaults(func=fetch_job_command)

    list_jobs = subparsers.add_parser("list-jobs", help="List submitted jobs")
    list_jobs.add_argument("--json", action="store_true", dest="as_json")
    list_jobs.add_argument("--metadata-root", default=None)
    list_jobs.set_defaults(func=list_jobs_command)

    connection = subparsers.add_parser(
        "connection",
        help="Manage private SSH connection profiles",
    )
    connection_commands = connection.add_subparsers(
        dest="connection_command",
        required=True,
    )

    connection_set = connection_commands.add_parser(
        "set",
        help="Create or update a connection profile",
    )
    connection_set.add_argument("name")
    connection_set.add_argument(
        "--ssh-command",
        help="Full login command copied from AutoDL, quoted as one argument",
    )
    connection_set.add_argument("--host", default=None)
    connection_set.add_argument("--user", default=None)
    connection_set.add_argument("--port", type=int, default=None)
    connection_set.add_argument("--identity-file", default=None)
    connection_set.add_argument(
        "--password",
        action="store_true",
        help="Prompt for and store an SSH password in the private local profile",
    )
    connection_set.add_argument(
        "--clear-password",
        action="store_true",
        help="Remove the stored password",
    )
    connection_set.add_argument(
        "--clear-identity-file",
        action="store_true",
        help="Remove the configured SSH identity file",
    )
    connection_set.add_argument(
        "--accept-new-host-key",
        action="store_true",
        help="Use OpenSSH accept-new policy without disabling changed-key checks",
    )
    connection_set.add_argument("--store", default=None)
    connection_set.set_defaults(func=connection_set_command)

    connection_show = connection_commands.add_parser(
        "show",
        help="Show one connection profile",
    )
    connection_show.add_argument("name")
    connection_show.add_argument("--store", default=None)
    connection_show.set_defaults(func=connection_show_command)

    connection_list = connection_commands.add_parser(
        "list",
        help="List connection profiles",
    )
    connection_list.add_argument("--json", action="store_true", dest="as_json")
    connection_list.add_argument("--store", default=None)
    connection_list.set_defaults(func=connection_list_command)

    connection_test = connection_commands.add_parser(
        "test",
        help="Test an SSH connection profile",
    )
    connection_test.add_argument("name")
    connection_test.add_argument("--timeout", type=float, default=15.0)
    connection_test.add_argument("--store", default=None)
    connection_test.set_defaults(func=connection_test_command)

    connection_remove = connection_commands.add_parser(
        "remove",
        help="Remove a connection profile",
    )
    connection_remove.add_argument("name")
    connection_remove.add_argument("--store", default=None)
    connection_remove.set_defaults(func=connection_remove_command)

    paper = subparsers.add_parser(
        "paper",
        help="Scaffold, validate, and run paper reproduction studies",
    )
    paper_commands = paper.add_subparsers(dest="paper_command", required=True)

    paper_init = paper_commands.add_parser(
        "init", help="Create an isolated paper reproduction workspace"
    )
    paper_init.add_argument("paper_id")
    paper_init.add_argument("--title", required=True)
    paper_init.add_argument("--url", default=None)
    paper_init.add_argument("--author", action="append", default=[])
    paper_init.add_argument("--year", type=int, default=None)
    paper_init.add_argument("--paper-root", default="paper_reproductions")
    paper_init.set_defaults(func=paper_init_command)

    paper_list = paper_commands.add_parser("list", help="List paper workspaces")
    paper_list.add_argument("--json", action="store_true", dest="as_json")
    paper_list.add_argument("--paper-root", default="paper_reproductions")
    paper_list.set_defaults(func=paper_list_command)

    paper_info = paper_commands.add_parser("info", help="Show a paper manifest")
    paper_info.add_argument("paper_id")
    paper_info.add_argument("--paper-root", default="paper_reproductions")
    paper_info.set_defaults(func=paper_info_command)

    paper_validate = paper_commands.add_parser(
        "validate", help="Validate paper claims, suites, recipes, and configs"
    )
    paper_validate.add_argument("paper_id")
    paper_validate.add_argument("--paper-root", default="paper_reproductions")
    paper_validate.set_defaults(func=paper_validate_command)

    paper_study = paper_commands.add_parser(
        "study", help="Run one reproduction recipe suite"
    )
    paper_study.add_argument("paper_id")
    paper_study.add_argument("--suite", default="smoke")
    paper_study.add_argument("--executor", default=None)
    paper_study.add_argument(
        "--set",
        action="append",
        default=[],
        dest="override",
        help="Apply key=value overrides to every recipe",
    )
    paper_study.add_argument("--detach", action="store_true")
    paper_study.add_argument("--poll-interval", type=float, default=1.0)
    paper_study.add_argument("--timeout", type=float, default=None)
    paper_study.add_argument("--paper-root", default="paper_reproductions")
    paper_study.add_argument("--study-root", default=None)
    paper_study.add_argument("--metadata-root", default=None)
    paper_study.set_defaults(func=paper_study_command)

    paper_assess = paper_commands.add_parser(
        "assess", help="Refresh and assess a submitted reproduction study"
    )
    paper_assess.add_argument("paper_id")
    paper_assess.add_argument("study_id")
    paper_assess.add_argument("--wait", action="store_true")
    paper_assess.add_argument("--poll-interval", type=float, default=1.0)
    paper_assess.add_argument("--timeout", type=float, default=None)
    paper_assess.add_argument("--paper-root", default="paper_reproductions")
    paper_assess.add_argument("--study-root", default=None)
    paper_assess.add_argument("--metadata-root", default=None)
    paper_assess.set_defaults(func=paper_assess_command)

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
