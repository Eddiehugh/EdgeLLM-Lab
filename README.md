# EdgeLLM-Lab

[中文说明](README.zh-CN.md)

EdgeLLM-Lab is a personal LLM algorithm learning, experimentation, and tool-building framework for edge-side and small-model research.

The project is organized around three levels:

```text
Level 1: Learn by Implementing
  Implement core LLM algorithms yourself to understand the principles.

Level 2: Experiment by Comparing
  Compare algorithms, model structures, compression methods, and runtimes in one reproducible framework.

Level 3: Work by Wrapping
  Wrap mature open-source projects as reusable tools without mixing their source code into the core system.
```

The core principle:

```text
Learn one technique -> implement one module -> run one experiment -> write one report.
```

## System Design

```text
EdgeLLM-Lab/
├── core/                    # Registry, component specs, config, provenance, runtime utilities
├── modules/                 # Level 1: Algorithm packages; each technique has its own file
├── models/                  # Level 1: TinyGPT, LLaMA-like, DeepSeek-like, future model families
├── training/                # Level 1/2: Losses, optimizers, schedulers, training entry points
├── inference/               # Level 1/2: Samplers, KV cache, generation engine
├── compression/             # Level 1/2: Quantization, pruning, low-rank compression
├── data/                    # Tokenizer, dataset, dataloader
├── experiments/             # Level 2: Registered stage pipeline, context, and run artifacts
├── execution/               # Provider-neutral job, runtime, artifact, and metadata control plane
├── reproduction/            # Level 2: Paper manifests, recipe suites, claim evaluation, reports
├── paper_reproductions/     # Isolated per-paper code, recipes, tests, and notes
├── benchmark/               # Level 2: Benchmark registries and metric collectors
├── backend/                 # Level 3: Runtime boundaries such as Torch, llama.cpp, ONNX, MLC
├── integrations/            # Level 3: Thin adapters for external open-source projects
├── external_projects/       # External project checkouts; not part of the core package
├── configs/                 # Experiment configs
├── docs/                    # Architecture and integration guides
├── tests/                   # Unit tests and manual debug probes
├── reports/                 # Learning notes and experiment reports
├── deploy/                  # Edge deployment experiments
├── cli.py                   # CLI entry point
└── main.py                  # Compatibility CLI entry point
```

## Level 1: Learn by Implementing

This level is for understanding LLM internals by writing simplified implementations yourself.

`modules/` uses a package-per-domain and file-per-technique layout:

```text
modules/
├── attention/
│   ├── mha.py
│   ├── mqa.py
│   ├── gqa.py
│   ├── mla.py
│   ├── sliding_window.py
│   └── sparse.py
├── mlp/
│   ├── gelu.py
│   └── swiglu.py
├── norm/
│   ├── layernorm.py
│   └── rmsnorm.py
├── block/
│   └── transformer.py
├── position/
│   └── rope.py
└── moe/
    └── router.py
```

`tests/` is separate from usable modules:

```text
tests/
├── unit/     # stable unit and smoke tests
└── debug/    # manually runnable debug probes
```

Current replaceable components:

- `attention`: MHA, MQA, GQA, learning-oriented MLA, sliding-window, top-k sparse attention.
- `mlp`: GELU MLP and SwiGLU.
- `norm`: LayerNorm and RMSNorm.
- `block`: Transformer block.
- `position_encoding`: RoPE.
- `model`: TinyGPT now; later LLaMA-like, SmolLM-like, MobileLLM-like, and multimodal families.
- `loss`: causal LM cross entropy, z-loss, distillation.
- `sampler`: greedy, multinomial, top-k, top-p.
- `kv_cache`: append-only cache plus layout-agnostic reference KV quantization; paged and sliding caches remain planned.
- `quantizer`: verified symmetric INT8 and packed group-wise INT4 reference implementations.
- `pruner`: local/global magnitude, structured channel, and N:M semi-structured pruning.
- `model I/O`: keyword-based tensor dictionaries and generic logits extraction, ready for text, vision, audio, and multimodal batches.

Example: add a custom attention implementation.

```python
from modules.attention import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("my_attention")
class MyAttention:
    ...
```

Use it from config:

```yaml
model:
  attention_type: my_attention
```

Registrations carry Level, maturity, capabilities, requirements, aliases, and
source metadata. Inspect them without constructing a model:

```bash
python3 -m cli list-components --details
python3 -m cli component-info backend llama_cpp
```

## Level 2: Experiment by Comparing

This level makes different algorithms comparable under the same framework.

Experiment flow:

```text
config.yaml
  -> registered pipeline stages
  -> build / train / evaluate / compress / export / benchmark
  -> collect metrics and provenance
  -> write run artifacts
```

The default pipeline preserves the original training behavior. A custom
extension can register a new `ExperimentStage` and place it in
`pipeline.stages`; `ExperimentRunner` does not need to change.

Each run writes:

```text
runs/<run-id>/
├── config.yaml
├── metrics.json
├── manifest.json
└── report.md
```

`manifest.json` records status, environment, Git revision, selected component
specifications, stage timings, errors, and artifact paths.

Quick smoke test:

```bash
python3 -m cli smoke
```

Run from config:

```bash
python3 -m cli train -c configs/smoke.yaml
```

Validate and resolve a config without executing it:

```bash
python3 -m cli validate-config -c configs/smoke.yaml
python3 -m cli validate-config -c configs/smoke.yaml --resolved
```

List registered components:

```bash
python3 -m cli list-components
```

Run stable tests:

```bash
python3 -m unittest discover -s tests
```

Run a manual debug probe:

```bash
python3 -m tests.debug.attention_variants_debug
```

Important Level 2 metrics:

- training loss and perplexity
- model size
- parameter count
- prefill latency
- decode latency
- TTFT and TPOT
- tokens/s
- peak memory
- KV cache memory
- quantization error
- effective sparsity and selected-weight storage
- backend runtime latency

### Quantization and pruning

Compression uses four separate boundaries: tensor algorithms, architecture-neutral
module selection, model transforms, and optimized backend/export adapters. The
current `ReferenceQuantizedLinear` is numerically testable and checkpointable,
but deliberately does not claim kernel speedup. Dense masked pruning likewise
reports sparsity without claiming reduced storage or latency.

Run the reference prune-then-quantize pipeline:

```bash
python3 -m cli train -c configs/compression/tiny_gpt_prune_int4.yaml
```

Models may expose `compression_scopes()` so language, vision, projector, and
other subgraphs can be compressed independently. See
[Compression Architecture](docs/COMPRESSION.md) for supported algorithms,
configuration, benchmark rules, backend boundaries, and the multimodal plan.

### Local and remote execution

The same experiment config can run locally or through SSH/AutoDL, Hugging Face
Jobs, ClearML, or a generated Colab notebook. Cloud adapters do not change the
experiment pipeline.

```bash
python3 -m cli submit -c configs/execution/local.yaml --wait
python3 -m cli list-jobs
python3 -m cli status <job-id>
python3 -m cli logs <job-id>
python3 -m cli fetch <job-id>
```

See [Remote Execution](docs/REMOTE_EXECUTION.md) for the architecture, provider
capability matrix, credentials, and configuration templates.
See [nanochat on AutoDL](docs/NANOCHAT_AUTODL.md) for the first local-edit to
cloud-training acceptance workflow.

### Paper reproduction

Turn a paper into isolated baseline/proposed recipes and machine-checkable claims:

```bash
python3 -m cli paper init 2405.00001 --title "Paper Title" --url <paper-url>
python3 -m cli paper validate 2405.00001
python3 -m cli paper study 2405.00001 --suite smoke
```

Paper-specific code stays under `paper_reproductions/<paper-id>/`; upstream
repositories still stay under `external_projects/`. See
[Paper Reproduction](docs/PAPER_REPRODUCTION.md).

## Level 3: Work by Wrapping

This level wraps mature projects as reusable tools while keeping their code outside the core framework.

Boundary rule:

```text
external_projects/<project>/repo
  External upstream source code and large artifacts.
  Not imported by the core package.

integrations/<project>/
  Thin local adapter, metadata, config mapping, conversion glue.

backend/
  Runtime boundary when an external project provides inference capability.
```

Known integration targets:

| Project | Fastest Use | Long-Term Role |
| --- | --- | --- |
| nanochat | Full-stack cloud training loop | External workloads, recipes, and checkpoint comparison |
| nanoGPT | Reference for a minimal train/generate loop | Absorb simple closed-loop ideas |
| TinyLlama | LLaMA-like architecture reference | Map configs and checkpoint structure |
| SmolLM | Small model family baseline | Compare small-model scales and recipes |
| MobileLLM | Edge architecture design reference | Absorb mobile-oriented model ideas |
| llama.cpp | Mature edge inference runtime | GGUF, quantized inference, backend benchmark |

Inspect integrations:

```bash
python3 -m cli list-integrations
python3 -m cli integration-info llama_cpp
python3 -m cli integration-info nanochat --local-path /path/to/nanochat
```

## External Project Policy

Open-source projects should not be vendored into the internal framework.

Use:

```text
external_projects/
├── nanochat/
├── tinyllama/repo/
├── smollm/repo/
├── mobilellm/repo/
└── llama_cpp/repo/
```

Do not import external project source from:

- `core/`
- `experiments/`
- `modules/`
- `models/`
- `training/`

Adapters should translate external project concepts into local registries, configs, checkpoints, benchmark targets, or backend calls.

## Development Roadmap

```text
v0.1: TinyGPT + MHA + train + generate
v0.2: RoPE + RMSNorm + SwiGLU
v0.3: KV cache + streaming generation
v0.4: MQA / GQA
v0.5: reference INT8 / packed INT4 / pruning pipeline
v0.6: MLA
v0.7: Sliding Window / Sparse Attention
v0.8: Benchmark suite
v0.9: llama.cpp / ONNX backend
v1.0: edge deployment demo
v1.1: multimodal model/data contracts and modality-aware compression
```

## Validation

```bash
python3 -m compileall core modules models training inference backend compression data experiments benchmark integrations cli.py main.py
python3 -m unittest discover -s tests
python3 -m cli list-components
python3 -m cli list-integrations
python3 -m cli train -c configs/smoke.yaml
```

## Design Documents

- [Architecture](docs/ARCHITECTURE.md)
- [Compression Architecture](docs/COMPRESSION.md)
- [Multimodal Architecture Plan](docs/MULTIMODAL.md)
- [Extension Guide](docs/EXTENDING.md)
- [Optimizer Architecture](docs/OPTIMIZERS.md)
- [Open Source Integration Workflow](docs/OPEN_SOURCE_INTEGRATION.md)
