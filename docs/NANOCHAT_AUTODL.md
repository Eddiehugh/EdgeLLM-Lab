# nanochat on AutoDL

This workflow validates local nanochat changes against cloud training while keeping
nanochat as an independent Git repository under `external_projects/nanochat`.

EdgeLLM-Lab pins both its own control-plane revision and the nanochat revision. It
executes structured argv commands, streams provider logs, and collects only declared
artifacts. It does not import or copy nanochat source into the framework.

## Run the baseline

Install `sshpass`, then configure password authentication whenever AutoDL changes the
endpoint:

```bash
brew install sshpass
python3 -m cli connection set autodl-main \
  --ssh-command "ssh -p <PORT> root@<HOST>" \
  --password --clear-identity-file \
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

The smoke recipe targets a prepared AutoDL instance. It does not install dependencies,
create a virtual environment, download data, or train a tokenizer. It uses the existing
environment at `/root/autodl-tmp/nanochat/.venv` together with the data and tokenizer
under `/root/autodl-tmp/nanochat`, then runs a 20-step d4 pretraining smoke. Only the
report, provenance files, and selected checkpoint are collected.

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
