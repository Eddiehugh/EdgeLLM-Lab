# Paper Reproductions

Each paper owns an isolated reproduction contract:

```text
paper_reproductions/<paper-id>/
├── paper.yaml             # metadata, claims, acceptance rules, suites
├── recipes/               # baseline/proposed and smoke/full experiment configs
├── implementation/        # only paper-specific components or stages
├── tests/                 # correctness tests for the implementation
└── notes/                 # reading notes and reproduction differences
```

Do not copy an upstream repository here. Keep upstream source under
`external_projects/` and pin its repository/revision in `paper.yaml`.

```bash
python3 -m cli paper init my-paper --title "Paper title"
python3 -m cli paper validate my-paper
python3 -m cli paper study my-paper --suite smoke
```
