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
├── core/                    # Registry, config loading, extension loading, runtime utilities
├── modules/                 # Level 1: Attention, MLP, Norm, RoPE, Block, MoE
├── models/                  # Level 1: TinyGPT, LLaMA-like, DeepSeek-like, future model families
├── training/                # Level 1/2: Losses, optimizers, schedulers, training entry points
├── inference/               # Level 1/2: Samplers, KV cache, generation engine
├── compression/             # Level 1/2: Quantization, pruning, low-rank compression
├── data/                    # Tokenizer, dataset, dataloader
├── experiments/             # Level 2: Config-driven experiment runner and run artifacts
├── benchmark/               # Level 2: Benchmark registries and metric collectors
├── backend/                 # Level 3: Runtime boundaries such as Torch, llama.cpp, ONNX, MLC
├── integrations/            # Level 3: Thin adapters for external open-source projects
├── external_projects/       # External project checkouts; not part of the core package
├── configs/                 # Experiment configs
├── docs/                    # Architecture and integration guides
├── reports/                 # Learning notes and experiment reports
├── deploy/                  # Edge deployment experiments
├── cli.py                   # CLI entry point
└── main.py                  # Compatibility CLI entry point
```

## Level 1: Learn by Implementing

This level is for understanding LLM internals by writing simplified implementations yourself.

Current replaceable components:

- `attention`: MHA now; later MQA, GQA, MLA, sliding-window, sparse attention.
- `mlp`: GELU MLP and SwiGLU.
- `norm`: LayerNorm and RMSNorm.
- `block`: Transformer block.
- `position_encoding`: RoPE.
- `model`: TinyGPT now; later LLaMA-like, SmolLM-like, MobileLLM-like.
- `loss`: causal LM cross entropy, z-loss, distillation.
- `sampler`: greedy, multinomial, top-k, top-p.
- `kv_cache`: append-only cache now; later paged, sliding, quantized cache.
- `quantizer`: INT8 now; later INT4, AWQ, GPTQ, KV quantization.

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

## Level 2: Experiment by Comparing

This level makes different algorithms comparable under the same framework.

Experiment flow:

```text
config.yaml
  -> build tokenizer / dataset / model / loss / optimizer / scheduler
  -> train or evaluate
  -> collect metrics
  -> write run artifacts
```

Each run writes:

```text
runs/<run-id>/
├── config.yaml
├── metrics.json
└── report.md
```

Quick smoke test:

```bash
python3 -m cli smoke
```

Run from config:

```bash
python3 -m cli train -c configs/smoke.yaml
```

List registered components:

```bash
python3 -m cli list-components
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
- backend runtime latency

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
| nanoGPT | Reference for a minimal train/generate loop | Absorb simple closed-loop ideas |
| TinyLlama | LLaMA-like architecture reference | Map configs and checkpoint structure |
| SmolLM | Small model family baseline | Compare small-model scales and recipes |
| MobileLLM | Edge architecture design reference | Absorb mobile-oriented model ideas |
| llama.cpp | Mature edge inference runtime | GGUF, quantized inference, backend benchmark |

Inspect integrations:

```bash
python3 -m cli list-integrations
python3 -m cli integration-info llama_cpp
python3 -m cli integration-info nanogpt --local-path /path/to/nanoGPT
```

## External Project Policy

Open-source projects should not be vendored into the internal framework.

Use:

```text
external_projects/
├── nanogpt/repo/
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
v0.5: INT8 / INT4 QuantLinear
v0.6: MLA
v0.7: Sliding Window / Sparse Attention
v0.8: Benchmark suite
v0.9: llama.cpp / ONNX backend
v1.0: edge deployment demo
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
- [Open Source Integration Workflow](docs/OPEN_SOURCE_INTEGRATION.md)
