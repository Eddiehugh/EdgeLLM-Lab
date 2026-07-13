# Paper Reproduction

The reproduction layer turns a paper into an executable and auditable study without creating a second training framework. It reuses registered components, experiment pipelines, remote executors, artifact stores, and reports.

```bash
python3 -m cli paper init 2405.00001 \
  --title "Paper Title" \
  --url https://arxiv.org/abs/2405.00001
python3 -m cli paper validate 2405.00001
python3 -m cli paper study 2405.00001 --suite smoke
```

Each workspace separates the contract from the implementation:

```text
paper_reproductions/<paper-id>/
├── paper.yaml             # metadata, claims, acceptance rules, suites
├── recipes/               # baseline/proposed and smoke/full configs
├── implementation/        # only paper-specific components or stages
├── tests/                 # implementation correctness tests
└── notes/                 # reading notes and reproduction differences
```

Claims support absolute targets and baseline comparisons using ratio, delta, or percent change. A claim with no executable expectations is reported as `NOT ASSESSED`, not as reproduced.

Recipes inherit normal experiment configs and contain only paper-specific differences. Standalone implementation files can be loaded through `extensions.paths` and register normal components or stages.

Use smoke suites for correctness and integration, ablation suites for mechanisms, full suites for paper-scale evidence, and edge suites for target-device latency, memory, energy, and export compatibility. Remote suites use the existing execution control plane:

```bash
python3 -m cli paper study 2405.00001 \
  --suite full \
  --executor huggingface_jobs \
  --detach
python3 -m cli paper assess 2405.00001 <study-id> --wait
```

Upstream repositories remain under `external_projects/`. See the detailed [Chinese guide](PAPER_REPRODUCTION.zh-CN.md) for manifest, recipe, expectation, and reporting examples.
