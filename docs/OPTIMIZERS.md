# Optimizer Architecture

Optimizer support is split into independent extension axes so algorithm
experiments do not become coupled to parameter selection or backend kernels.

```text
training/optimizers/
├── api.py                 config normalization and construction
├── registry.py            implementation and policy registries
├── reference/             readable Level 1 algorithms
├── adapters/              mature Level 3 library backends
└── policies/              model parameter grouping rules
```

The built-ins are:

- `reference_adamw`: readable AdamW equations for learning and parity testing.
- `torch_adamw` / `adamw`: production PyTorch AdamW.
- `torch_sgd` / `sgd`: production PyTorch SGD.
- `all`: one parameter group, compatible with anonymous parameter iterables.
- `decay_by_dimension`: decay matrices and exclude vectors/scalars from decay.

## Configuration

Use the structured form for new experiments:

```yaml
training:
  optimizer:
    algorithm: adamw
    implementation: torch
    param_group_policy:
      type: decay_by_dimension
      min_decay_ndim: 2
    lr: 0.0003
    betas: [0.9, 0.95]
    weight_decay: 0.1
```

`algorithm` and `implementation` resolve to `<implementation>_<algorithm>`.
The legacy `{type: adamw}` form remains supported and resolves to
`torch_adamw`.

Parameter policies other than `all` require `model=` because anonymous
`model.parameters()` iterators discard names and structural context.

## Adding Implementations

Add a readable algorithm under `reference/` when the update rule itself is the
research target. Add a thin adapter under `adapters/` for stable packaged
dependencies such as PyTorch, bitsandbytes, or DeepSpeed. An external source
checkout such as nanochat remains under `external_projects/`; its integration
may register an optimizer through the extension mechanism without importing
that checkout into the core package. Verify optimized variants against the
readable implementation in `tests/parity/optimizers/`.

Add parameter routing independently under `policies/`. Composite optimizers,
such as matrix parameters using Muon and embeddings/scalars using AdamW, should
be implemented as composition over selectors and optimizer implementations;
they should not embed parameter-name rules inside the Muon update equations.
