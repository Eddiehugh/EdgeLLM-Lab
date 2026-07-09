# External Project Manifests

Use this folder for small YAML metadata files that describe local checkouts.

Example:

```yaml
name: nanogpt
repo_path: external_projects/nanogpt/repo
upstream: https://github.com/karpathy/nanoGPT
mode: reference
notes:
  - Used only as a reference implementation.
```

Do not store cloned source code in this folder.
