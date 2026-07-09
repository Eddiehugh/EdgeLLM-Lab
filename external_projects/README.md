# External Projects Workspace

This directory is reserved for open-source project checkouts and other external
artifacts.

It is intentionally not a Python package and should not be imported by core
code. The main system talks to external projects through thin adapters under
`integrations/<project>/`.

Recommended layout:

```text
external_projects/
├── nanogpt/
│   ├── repo/          # git clone of the upstream project
│   ├── notes.md       # local notes
│   ├── patches/       # optional local patches
│   └── artifacts/     # optional generated outputs
├── tinyllama/
│   └── repo/
└── llama_cpp/
    └── repo/
```

Default adapter lookup path:

```text
external_projects/<project>/repo
```

You can also point an adapter at any local path:

```bash
python3 -m cli integration-info nanogpt --local-path /path/to/nanoGPT
```

Boundary rule:

- `external_projects/`: external source, upstream scripts, large generated files.
- `integrations/`: small local adapters, metadata, config mapping, conversion glue.
- `core/`, `experiments/`, `modules/`, `models/`: internal system code only.
