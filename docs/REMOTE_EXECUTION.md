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

Every automated backend invokes the same `execution.worker`, which calls the existing `ExperimentRunner` and publishes its complete run directory. Provider code never enters the algorithm or experiment pipeline layers.

## Providers

| Provider | Lifecycle | Artifact requirement |
| --- | --- | --- |
| Local | Fully automated | Local, S3, or Hub |
| SSH / AutoDL | Fully automated with system SSH/SCP | Remote local, S3, or Hub |
| Hugging Face Jobs | Official Jobs API | S3 or Hub |
| ClearML | Task + Agent queue | S3 or Hub |
| Colab | Reproducible notebook generation | Manual local result or remote store |

Colab is deliberately prepare-only because it does not expose a stable general-purpose submit/status/cancel API. AutoDL is an SSH profile rather than a vendor-specific protocol.

Remote jobs default to a clean-worktree requirement and a pinned Git revision. Configuration files contain secret variable names, never secret values. See the runnable templates under `configs/execution/` and the detailed [Chinese guide](REMOTE_EXECUTION.zh-CN.md).

Install only the provider dependencies you use:

```bash
pip install -e '.[hf]'
pip install -e '.[clearml]'
pip install -e '.[s3]'
pip install -e '.[cloud]'
```
