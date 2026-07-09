# Integrations

External projects should be wrapped here instead of copied into core training or
inference code.

This directory contains only thin local adapters. Upstream source checkouts
belong in `external_projects/<project>/repo` or another external workspace.

Recommended adapters:

- `nanogpt/`: minimal train/generate loop references and checkpoint conversion.
- `tinyllama/`: LLaMA-like architecture configs and checkpoint mapping.
- `smollm/`: small model family configs and benchmark comparisons.
- `mobilellm/`: edge-oriented architecture experiments.
- `llama_cpp/`: GGUF export, quantized inference, and backend benchmarking.

Each adapter should expose local components through registries, config
templates, conversion glue, or `backend/` implementations.

Use:

```bash
python3 -m cli list-integrations
python3 -m cli integration-info nanogpt --templates
```

See `docs/OPEN_SOURCE_INTEGRATION.md` for the full workflow.
