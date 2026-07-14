# Remote Execution Control Plane

Phase 1 separates experiment semantics from compute providers. Models, replaceable components, pipelines, and reports stay inside EdgeLLM-Lab; providers only supply compute.

```text
CLI -> RunManager
       ├── Executor: Local, SSH/AutoDL, Hugging Face Jobs, ClearML, Colab prepare
       ├── Runtime: Native, Docker
       ├── ArtifactStore: Local, S3, Hugging Face Hub
       └── MetadataStore: JSON
```

Executors own submit/status/logs/cancel/fetch. Runtimes own process isolation. Artifact stores own checkpoints and reports. Metadata stores own small lifecycle records. These interfaces are intentionally independent.

## Commands

```bash
python3 -m cli submit -c configs/execution/local.yaml --wait
python3 -m cli list-jobs
python3 -m cli status <job-id>
python3 -m cli logs <job-id> --tail 100
python3 -m cli cancel <job-id>
python3 -m cli fetch <job-id> --output downloads/<job-id>
```

Every automated backend invokes the same `execution.worker`. Internal workloads call
the existing `ExperimentRunner`; external workloads checkout a pinned Git revision,
execute structured argv commands, and collect only declared artifacts. Provider code
never enters the algorithm or experiment pipeline layers.

## Providers

| Provider | Lifecycle | Artifact requirement |
| --- | --- | --- |
| Local | Fully automated | Local, S3, or Hub |
| SSH / AutoDL | Fully automated with system SSH/SCP | Remote local, S3, or Hub |
| Hugging Face Jobs | Official Jobs API | S3 or Hub |
| ClearML | Task + Agent queue | S3 or Hub |
| Colab | Reproducible notebook generation | Manual local result or remote store |

Colab is deliberately prepare-only because it does not expose a stable general-purpose submit/status/cancel API. AutoDL is an SSH profile rather than a vendor-specific protocol.

Keep changing AutoDL SSH endpoints in the private local connection store rather
than experiment configs:

```bash
python3 -m cli connection set autodl-main \
  --ssh-command "ssh -p 35394 root@region-1.autodl.com" \
  --identity-file ~/.ssh/id_ed25519 \
  --accept-new-host-key
python3 -m cli connection test autodl-main
```

Configs reference `execution.executor.profile: autodl-main`. Profiles are kept
in the gitignored `.edgellm/connections.json` with mode `600` and are resolved
into each immutable JobSpec at submission time.

AutoPanel transfers files between Quark and `/root/autodl-tmp`; Quark is not a
mounted filesystem. Use the instance data disk for active training, optional
AutoDL File Storage at `/root/autodl-fs` for continuously durable artifacts,
and Quark as a cold backup. See the Chinese guide for the full migration flow.

Remote jobs default to a clean-worktree requirement and a pinned Git revision. Configuration files contain secret variable names, never secret values. See the runnable templates under `configs/execution/` and the detailed [Chinese guide](REMOTE_EXECUTION.zh-CN.md).

Install only the provider dependencies you use:

```bash
pip install -e '.[hf]'
pip install -e '.[clearml]'
pip install -e '.[s3]'
pip install -e '.[cloud]'
```
