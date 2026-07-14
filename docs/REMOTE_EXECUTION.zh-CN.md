# 远端执行控制面

第一阶段解决的是“本机无法训练，但实验定义不能因云平台而改变”。算法组件、模型、Pipeline 和报告仍由 EdgeLLM-Lab 自己管理；云平台只负责提供计算资源。

## 分层

```text
CLI
└── RunManager
    ├── Executor
    │   ├── LocalExecutor
    │   ├── SSHExecutor (AutoDL profile)
    │   ├── HuggingFaceJobsExecutor
    │   ├── ClearMLExecutor
    │   └── ColabExecutor (prepare-only)
    ├── Runtime
    │   ├── NativeRuntime
    │   └── DockerRuntime
    ├── ArtifactStore
    │   ├── LocalArtifactStore
    │   ├── S3ArtifactStore
    │   └── HuggingFaceHubArtifactStore
    └── MetadataStore
        └── JsonMetadataStore
```

这四个接口不能合并：Executor 管作业生命周期，Runtime 管进程隔离，ArtifactStore 管大文件，MetadataStore 管小型控制面记录。实验 Runner 不感知 AutoDL、HF Jobs 或 ClearML。

## 能力矩阵

| 后端 | submit | status | logs | cancel | fetch | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| Local | 自动 | 自动 | 自动 | 自动 | 自动 | 后台子进程 |
| SSH / AutoDL | 自动 | 自动 | 自动 | 自动 | 自动 | 复用系统 `ssh`/`scp` |
| Hugging Face Jobs | 自动 | 自动 | 自动 | 自动 | 自动 | 使用 `huggingface_hub` 官方 Jobs API |
| ClearML | 自动 | 自动 | 自动 | 自动 | 自动 | 使用 Task + Agent queue |
| Colab | 生成 Notebook | 手动 | 手动 | 本地标记 | 远端 Store | Colab 没有稳定的通用作业生命周期 API |

## 作业生命周期

每次提交会生成一个不可变 `JobSpec`，包含已展开的实验配置、Git revision、Executor、Runtime、ArtifactStore 和非敏感环境变量。控制面只在 `.edgellm/jobs/<job-id>/record.json` 保存状态，不把模型与 checkpoint 塞入元数据。

```bash
python3 -m cli submit -c configs/execution/local.yaml
python3 -m cli submit -c configs/execution/local.yaml --wait
python3 -m cli list-jobs
python3 -m cli status <job-id>
python3 -m cli logs <job-id> --tail 100
python3 -m cli cancel <job-id>
python3 -m cli fetch <job-id> --output downloads/<job-id>
```

所有自动后端最终运行同一个入口：

```bash
python3 -m execution.worker
```

Worker 根据 `WorkloadSpec` 分发任务：内置实验调用 `ExperimentRunner`；独立开源项目
从固定 Git revision checkout 后执行结构化 setup/command，并只收集声明的产物路径。
两种模式都生成 `metrics.json`、`manifest.json` 和 `report.md`，但不会强迫外部项目
采用 EdgeLLM-Lab 的内部模型接口。

## 后端配置

- Local：`configs/execution/local.yaml`
- AutoDL：`configs/execution/autodl.yaml`
- AutoDL + nanochat：`configs/execution/autodl_nanochat_smoke.yaml`
- Hugging Face Jobs：`configs/execution/huggingface_jobs.yaml`
- ClearML：`configs/execution/clearml.yaml`
- Colab：`configs/execution/colab.yaml`

AutoDL 没有必要单独发明协议。它是带持久磁盘的 SSH 机器，因此复用 `SSHExecutor`，默认工作目录是 `/root/autodl-tmp/edgellm-jobs`。

### AutoDL 动态 SSH 连接

实验配置不保存会随实例变化的 host 和 port，只引用本机私有 Profile：

```yaml
execution:
  executor:
    type: autodl
    profile: autodl-main
    remote_root: /root/autodl-tmp/edgellm-jobs
```

第一次创建实例，或更换实例后，直接粘贴 AutoDL 显示的 SSH 命令：

```bash
python3 -m cli connection set autodl-main \
  --ssh-command "ssh -p 35394 root@region-1.autodl.com" \
  --identity-file ~/.ssh/id_ed25519 \
  --accept-new-host-key

python3 -m cli connection test autodl-main
python3 -m cli connection show autodl-main
python3 -m cli connection list
```

再次换机时重复 `connection set` 即可，所有引用 `autodl-main` 的实验配置无需修改。
连接信息保存在 `.edgellm/connections.json`，该目录已被 Git 忽略，文件权限为
`600`。Profile 不保存 SSH 密码；应为 AutoDL 实例配置公钥认证。

提交作业时 Profile 会被解析并固化进 `JobSpec`。之后更新 Profile 只影响新作业，
不会把历史作业的 status、logs 或 fetch 请求误发到新机器。

### AutoDL 数据分层与夸克备份

夸克网盘由 AutoPanel 负责上传和下载，不是 Linux 挂载目录。不能把
`remote_root` 或训练中的 checkpoint 路径直接设置为“夸克路径”。推荐分三层：

```text
/root/autodl-tmp/       热数据：当前实例训练、数据集缓存、临时 checkpoint
/root/autodl-fs/        温数据：同地区新实例可挂载的文件存储，可选
夸克网盘                冷数据：数据集、最终 checkpoint、报告和环境清单备份
```

AutoDL 官方建议训练前把数据从网盘复制到 `/root/autodl-tmp`，避免网盘 IO 限制。
如果需要训练过程中持续抗实例释放，应启用 AutoDL 文件存储，并把 ArtifactStore
配置到挂载目录：

```yaml
execution:
  artifact_store:
    type: local
    root: /root/autodl-fs/EdgeLLM-Lab/artifacts
```

作业结束后，再通过 AutoPanel 把 artifact 目录上传到夸克。只使用夸克而不使用
文件存储时，上传和恢复是手动步骤；AutoDL 当前没有在公开文档中提供可供
Executor 调用的夸克上传 API，因此不能声称 checkpoint 已经被实时同步。

新实例恢复流程：

1. 使用 AutoPanel 从夸克下载数据和 artifact 到 `/root/autodl-tmp`。
2. 执行一次 `connection set autodl-main --ssh-command "..."` 更新连接。
3. 执行 `connection test autodl-main` 验证公钥、host 和 port。
4. 重新提交原实验配置，或从持久 checkpoint 恢复。

不要备份整个 Conda/venv 作为环境复现手段。代码由固定 Git revision 获取，Python
依赖应由 `pyproject.toml`/lock file 重建；夸克只保存难以重新生成的大文件和实验产物。

Hugging Face Jobs 和 ClearML 机器可能在任务结束后释放，所以禁止使用 `LocalArtifactStore`。必须配置 `huggingface_hub` 或 `s3`。SSH/AutoDL 可直接用 `scp` 取回结果，因此允许远端本地 Store。

Colab 后端只生成 `.ipynb`。Notebook 固定 Git revision、安装项目并运行统一 Worker；用户仍需在 Colab 页面选择硬件并启动。

## Git 与复现约束

远端提交默认要求 Git worktree 干净，因为远端只能获取已提交的 revision。公开 GitHub SSH URL 会自动转成 HTTPS URL，避免云容器缺少 GitHub SSH key。私有仓库应显式配置 `execution.source.repo_url` 和平台 secret。

仅在明确知道后果时关闭检查：

```yaml
execution:
  source:
    require_clean: false
```

关闭检查不会上传未提交代码，远端仍然运行配置中的 Git revision。

## 依赖与凭据

```bash
pip install -e '.[hf]'
pip install -e '.[clearml]'
pip install -e '.[s3]'
pip install -e '.[cloud]'
```

配置文件只写环境变量名称，不写 token 值。例如 HF Jobs 的 `secrets.HF_TOKEN: HF_TOKEN` 表示从提交机的 `HF_TOKEN` 读取值，并作为远端 secret 注入。S3 默认使用 AWS SDK/fsspec 的标准凭据链。ClearML secret 应配置在 Agent 环境或 ClearML Vault。

## 第一阶段边界

当前阶段不负责自动创建/停止 AutoDL 实例，不提供云资源价格调度，不做多节点训练编排，也不把 Colab 包装成不存在的全自动 Job API。这些能力可以后续通过新的 Executor 或 Scheduler 增加，不需要修改算法层与 ExperimentRunner。
