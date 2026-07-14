# Open Source Integration Workflow

The fastest way to reuse an open-source LLM project is not to copy it into this
repo. Use a staged integration path.

## Three Modes

### 1. Reference

Use the external project as a runnable reference.

Best for:

- quickly reproducing a result
- understanding the original training or inference flow
- comparing your local implementation against a known baseline

Keep this mode outside the core code path. Store upstream checkouts, notes,
patches, and large artifacts under `external_projects/<project>/`.

### 2. Adapter

Wrap the external project behind local interfaces.

Best for:

- checkpoint conversion
- backend runtime calls
- benchmark comparison
- config translation

Adapters live under `integrations/<project>/` and expose capabilities through
local registries, configs, or backend classes. The experiment runner should
still call local factories.

### 3. Absorb

Port the core idea into local modules.

Best for:

- Attention variants
- model blocks
- loss functions
- quantization methods
- KV cache policies

After absorbing, the technique should be usable by config:

```yaml
model:
  attention_type: my_new_attention

loss:
  type: my_new_loss
```

## Project Mapping

| Project | Fastest Use | Long-Term Integration |
| --- | --- | --- |
| nanoGPT | Reference for minimal train/generate loop | Absorb simple loop ideas into `experiments/runner.py` |
| TinyLlama | LLaMA-like architecture reference | Map configs and port missing blocks locally |
| SmolLM | Small model family baseline | Compare model scales and benchmark results |
| MobileLLM | Edge architecture design reference | Absorb architecture ideas and benchmark edge metrics |
| llama.cpp | Mature edge inference runtime | Wrap through `backend/llama_cpp_backend.py` and export/benchmark |

## CLI

List known integrations:

```bash
python3 -m cli list-integrations
```

Inspect one integration:

```bash
python3 -m cli integration-info nanochat --templates
```

If you have a local checkout:

```bash
python3 -m cli integration-info nanochat --local-path /path/to/nanochat
```

Default checkout path:

```text
external_projects/<project>/repo
```

You can change the workspace root:

```bash
python3 -m cli list-integrations --external-root /Volumes/LLM-Projects
python3 -m cli integration-info llama_cpp --external-root /Volumes/LLM-Projects
```

## Adapter Checklist

When adding a new project:

1. Create `integrations/<project>/adapter.py`.
2. Register an `IntegrationAdapter`.
3. Document purpose, modes, capabilities, and first step.
4. Add config templates if useful.
5. Put upstream source under `external_projects/<project>/repo` or another external path.
6. Add conversion glue under `integrations/<project>/` only when it is small and local.
7. Keep large outputs under `external_projects/<project>/artifacts` or run directories.
8. Expose runtime behavior through `backend/` if it performs inference.
9. Expose model ideas through `modules/` or `models/` only if they are absorbed.
10. Add benchmark comparisons under `benchmark/`.

The key rule: external project details should stop at the adapter boundary.
