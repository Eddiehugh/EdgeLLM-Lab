# EdgeLLM-Lab Architecture

This project is designed to grow by adding components, not by rewriting training
or inference scripts.

## Stable Layers

```text
core/          generic framework utilities
experiments/   config-driven orchestration and run artifacts
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
domain and one file per technique.

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
├── unit/     stable automated tests
└── debug/    manually runnable debug probes
```

## Component Rule

Every replaceable part follows the same pattern:

1. Define a registry in the domain package.
2. Register implementations with a stable lowercase name.
3. Build from config through a `build_*` factory.
4. Let the experiment runner consume only factories, never concrete classes.

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
└── report.md
```

Future artifacts such as checkpoints, generated samples, benchmark CSV files,
profiles, and exported models should live under the same run directory.

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
├── nanogpt/repo/
├── tinyllama/repo/
├── smollm/repo/
├── mobilellm/repo/
└── llama_cpp/repo/

integrations/
├── nanogpt/adapter.py
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
