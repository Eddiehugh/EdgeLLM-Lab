# nanochat on AutoDL

This workflow validates local nanochat changes against cloud training while keeping
nanochat as an independent Git repository under `external_projects/nanochat`.

EdgeLLM-Lab pins both its own control-plane revision and the nanochat revision. It
executes structured argv commands, streams provider logs, and collects only declared
artifacts. It does not import or copy nanochat source into the framework.

## Run the baseline

Configure key-based SSH whenever AutoDL changes the endpoint:

```bash
python3 -m cli connection set autodl-main \
  --ssh-command "ssh -p <PORT> root@<HOST>" \
  --identity-file ~/.ssh/id_ed25519 \
  --accept-new-host-key
python3 -m cli connection test autodl-main
```

Both Git worktrees must be clean and their revisions must be available through their
configured remotes. Then submit and inspect the run:

```bash
python3 -m cli submit -c configs/execution/autodl_nanochat_smoke.yaml
python3 -m cli list-jobs
python3 -m cli logs <job-id> --tail 200
python3 -m cli status <job-id>
python3 -m cli fetch <job-id> --output downloads/<job-id>
```

The first run creates nanochat's uv environment, downloads one training shard plus the
validation shard, trains an 8192-token tokenizer, and runs a 20-step d4 pretraining
smoke. Dataset and tokenizer caches remain under `/root/autodl-tmp/edgellm-cache`;
only the report, provenance files, and selected checkpoint are collected.

## Modify locally, run remotely

Fork nanochat before making changes, point its `origin` at your fork, and use this loop:

```bash
cd external_projects/nanochat
git switch -c experiment/my-attention
# edit and run nanochat's own focused tests
git add <files>
git commit -m "experiment: change attention"
git push -u origin experiment/my-attention
cd ../..
python3 -m cli submit -c configs/execution/autodl_nanochat_smoke.yaml
```

Cloud submission rejects uncommitted changes by default. A successful acceptance run
ends in `completed` and fetches `external-run.json`, `metrics.json`, `report.md`, and
the `edgellm-smoke` checkpoint.
