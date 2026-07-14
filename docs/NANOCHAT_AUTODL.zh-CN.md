# nanochat AutoDL 闭环

本流程验证一件事：在本地独立修改 nanochat，通过 EdgeLLM-Lab 固定两个仓库的
revision，在 AutoDL 上训练，并从统一 CLI 查看日志和取回 checkpoint。

## 仓库边界

```text
EdgeLLM-Lab/                    控制面、配置、任务记录、产物收集
external_projects/nanochat/     独立 Git 仓库、独立依赖、独立上游历史
AutoDL job/source/              云端固定 revision 的 EdgeLLM-Lab
AutoDL job/projects/nanochat/   云端固定 revision 的 nanochat
```

EdgeLLM-Lab 不导入 nanochat 的 Python 模块。`WorkloadSpec` 只保存 Git source、
setup argv、train argv 和需要收集的产物路径。命令不经过 shell，未提交的本地代码
不会被隐式上传。

## 首次准备

1. 安装 `sshpass`：`brew install sshpass`。
2. 更新 endpoint，通过隐藏输入保存密码，并验证连接：

```bash
python3 -m cli connection set autodl-main \
  --ssh-command "ssh -p <PORT> root@<HOST>" \
  --password --clear-identity-file \
  --accept-new-host-key
python3 -m cli connection test autodl-main
```

3. 确认两个仓库都能被 AutoDL 通过 HTTPS 拉取，并且工作区干净：

```bash
git status --short
git -C external_projects/nanochat status --short
git -C external_projects/nanochat remote -v
```

当前官方 nanochat checkout 可以直接完成第一次基线运行。开始修改 nanochat 前，
应在 GitHub fork 它，并把本地 `origin` 改为自己的 fork；否则本地 commit 无法推送，
AutoDL 也无法取得修改后的 revision。

## 提交和观察

```bash
python3 -m cli submit -c configs/execution/autodl_nanochat_smoke.yaml
python3 -m cli list-jobs
python3 -m cli logs <job-id> --tail 200
python3 -m cli status <job-id>
python3 -m cli fetch <job-id> --output downloads/<job-id>
```

也可以在首轮直接等待完成：

```bash
python3 -m cli submit \
  -c configs/execution/autodl_nanochat_smoke.yaml \
  --wait --poll-interval 5 --timeout 7200
```

首次任务会安装 uv、创建 nanochat `.venv`、下载一个训练 shard 和固定的 validation
shard、训练 8192 词表 tokenizer，然后训练 20 step 的 d4 模型。数据与 tokenizer
留在 `/root/autodl-tmp/edgellm-cache`，后续任务复用；下载结果只包含报告、执行
manifest 和 `edgellm-smoke` checkpoint，不包含数据集。

## 本地修改到云端

```bash
cd external_projects/nanochat
git switch -c experiment/my-attention
# 修改并执行 nanochat 自己的测试
git add <files>
git commit -m "experiment: change attention"
git push -u origin experiment/my-attention
cd ../..

python3 -m cli submit -c configs/execution/autodl_nanochat_smoke.yaml
```

提交时 EdgeLLM-Lab 自动读取 nanochat 当前 commit。若 nanochat 或 EdgeLLM-Lab 有
未提交修改，默认拒绝云端任务。这个约束保证本地看到的代码与云端运行代码一致。

## 验收标准

- `connection test` 能通过本地 Profile 自动完成密码认证。
- 作业日志显示 CUDA GPU、nanochat revision 和 20 个训练 step。
- `status` 最终为 `completed`。
- 下载目录包含 `external-run.json`、`metrics.json`、`report.md` 和 checkpoint。
- 修改 nanochat 并提交后，新作业记录的是新的 nanochat revision。

若使用 AutoDL 文件存储，把 ArtifactStore root 改到
`/root/autodl-fs/EdgeLLM-Lab/artifacts`。只使用夸克时，作业结束后仍需通过 AutoPanel
把 artifact 和必要 cache 上传；夸克不是训练时挂载盘。
