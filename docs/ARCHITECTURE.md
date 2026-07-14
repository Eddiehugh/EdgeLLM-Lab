# EdgeLLM-Lab Architecture

This project is designed to grow by adding components, not by rewriting training
or inference scripts.

## Stable Layers

```text
core/          generic framework utilities
experiments/   config-driven orchestration and run artifacts
execution/     provider-neutral job lifecycle, runtime, artifact, and metadata control plane
reproduction/  paper contracts, recipe suites, claim evaluation, and study reports
data/          tokenizer, dataset, dataloader components
modules/       algorithm packages with one file per technique
models/        model families and model factories
training/      losses, optimizers, schedulers, trainer entry points
inference/     samplers, KV cache, generation engines
compression/   quantization, pruning, low-rank methods
backend/       runtime adapters such as torch, llama.cpp, ONNX, MLC
benchmark/     benchmark suites and metric collectors
integrations/  thin local adapters for external projects
external_projects/ independent workspace for external source checkouts
tests/         unit tests and manual debug probes
```

`core/` and `experiments/` should stay small and domain-agnostic. Algorithm code
belongs in domain packages.

## Module Layout Rule

Do not keep a whole technical family in one large file. Use one package per
domain and one file per simple technique. When a technique gains independent
configs, kernels, cache layouts, or conversion logic, promote that file to a
subpackage while preserving its public import path.

Example:

```text
modules/attention/
├── registry.py
├── base.py
├── mha.py
├── mqa.py
├── gqa.py
├── mla.py
├── sliding_window.py
└── sparse.py
```

The package `__init__.py` should only expose public imports and load built-ins.
The registry and factory live in `registry.py`. Shared utilities live in
`base.py`.

Tests are separate from usable modules:

```text
tests/
├── unit/     stable behavioral tests
├── parity/   numerical comparison with mature implementations
└── debug/    manually runnable debug probes
```

Optimizers have additional independent axes because an update equation, a
parameter routing rule, and a fused/distributed backend are different research
variables. Keep readable algorithms in `training/optimizers/reference/`,
mature implementations in `adapters/`, and parameter grouping in `policies/`.
See [Optimizer Architecture](OPTIMIZERS.md).

## Component Rule

Every replaceable part follows the same pattern:

1. Define a registry in the domain package.
2. Register implementations with a stable lowercase name.
3. Build from config through a `build_*` factory.
4. Let the experiment runner consume only factories, never concrete classes.

Registrations also expose a `ComponentSpec`: project Level, maturity,
capabilities, runtime requirements, aliases, and source module. Maturity uses
the explicit states `planned`, `experimental`, `verified`, and `production`.
An importable placeholder must be marked `planned`, not presented as supported.

Example:

```python
from modules.attention import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("my_attention")
class MyAttention:
    ...
```

Config:

```yaml
model:
  attention_type: my_attention
```

Use `python3 -m cli component-info attention my_attention` to inspect the
resolved metadata.

## Pipeline Rule

`ExperimentRunner` owns lifecycle and artifacts, but it does not build models
or contain training logic. Work is performed by registered `ExperimentStage`
classes through a shared `ExperimentContext`.

Each stage declares `requires` and `provides`. The pipeline validates dependency
order before execution and checks declared outputs after every stage. Built-in
stages live in `experiments/stages/`; custom stages can be loaded through the
same `imports` extension mechanism as algorithm modules.

```yaml
pipeline:
  stages:
    - runtime_setup
    - build_data
    - build_model
    - build_training
    - train
    - model_stats
    - checkpoint
```

Future evaluation, compression, export, deployment, and benchmark workflows
should be new stages or new stage sequences, not conditionals added to the
runner.

## Execution Rule

`ExperimentRunner` defines internal experiments. `WorkloadSpec` can instead pin
an independent external repository and structured commands. `RunManager` defines
where either workload runs. Provider SDKs may only appear in
`execution/executors/`; they must not be imported by algorithm modules,
experiment stages, or external projects.

Keep four independent extension points:

1. `Executor`: submit, status, logs, cancel, and provider-specific fetch.
2. `Runtime`: native process or container isolation.
3. `ArtifactStore`: durable checkpoints, metrics, reports, and exports.
4. `MetadataStore`: small job specifications and lifecycle records.

All automated providers invoke `execution.worker`, which dispatches either the
normal experiment runner or an external workload. Adding another cloud provider
therefore requires a new Executor, not another training entry point.

## Paper Reproduction Rule

A paper reproduction is a study over normal experiment recipes, not a new
runner. Keep framework code in `reproduction/` and per-paper content in
`paper_reproductions/<paper-id>/`.

Each paper must separate:

1. Metadata and claims in `paper.yaml`.
2. Baseline/proposed and smoke/full configurations in `recipes/`.
3. Only paper-specific components or stages in `implementation/`.
4. Correctness tests in `tests/` and methodological differences in `notes/`.

Paper claims need executable expectations before they can pass. Upstream
repositories remain in `external_projects/` and are referenced by immutable
revisions.

## Extension Rule

Custom research modules do not need to modify the core repo. Add import paths to
the experiment config:

```yaml
imports:
  - my_lab.attention
  - my_lab.losses
```

The runner imports those modules before building components, so decorators can
register new implementations.

## Experiment Rule

One experiment config should fully describe:

- model architecture
- data source and tokenizer
- training objective
- optimizer and scheduler
- inference or benchmark backend
- output directory

Every run writes:

```text
runs/<run-id>/
├── config.yaml
├── metrics.json
├── manifest.json
└── report.md
```

`manifest.json` is the machine-readable lifecycle record. It captures status,
environment, Git revision, component specs, stage timings, failures, and
artifact paths. Checkpoints, generated samples, benchmark files, profiles, and
exported models should live under the same run directory and be registered in
that manifest.

## Adapter Rule

External projects such as nanoGPT, TinyLlama, SmolLM, MobileLLM, and llama.cpp
must stay outside the internal code path. The repository has two separate
spaces:

- `external_projects/<project>/repo`: optional checkout of upstream source.
- `integrations/<project>/`: small local adapter metadata and glue code.

The adapter may point at the external checkout, but it must not make the
external project part of the internal model, training, or inference APIs.

Recommended layout:

```text
external_projects/
├── nanochat/
├── tinyllama/repo/
├── smollm/repo/
├── mobilellm/repo/
└── llama_cpp/repo/

integrations/
├── nanochat/adapter.py
├── tinyllama/adapter.py
├── smollm/adapter.py
├── mobilellm/adapter.py
└── llama_cpp/adapter.py
```

Adapters should translate external project concepts into local registries,
configs, checkpoints, or backend interfaces.

Use `python3 -m cli list-integrations` to inspect known adapter boundaries.

## Non-Goals

- Do not put one-off experiment logic into `core/`.
- Do not make `experiments/runner.py` depend on a specific model family.
- Do not special-case custom research components inside shared training code.
- Do not let a backend adapter change the model definition API.
- Do not import external project source from `core/`, `experiments/`, `modules/`, or `models/`.
- Do not vendor external repositories into installable Python packages.
