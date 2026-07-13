# 论文复现

论文复现层用于把“读到一个方法”快速转换成可执行、可比较、可审计的实验。它复用现有组件注册表、Experiment Pipeline、远端 Executor、ArtifactStore 和报告系统，不创建第二套训练框架。

## 推荐流程

```text
阅读论文
  -> 提取 claim 与原始证据位置
  -> 创建 paper workspace
  -> 编写最小 paper-specific component
  -> correctness test
  -> smoke suite
  -> scaled/full suite
  -> 自动评估 claim
  -> 记录复现差异与结论
```

先验证算法方向和实现正确性，再扩大数据、模型和训练步数。复现论文不等于第一次实验就完整复制论文算力。

## 快速开始

```bash
python3 -m cli paper init 2405.00001 \
  --title "Paper Title" \
  --url https://arxiv.org/abs/2405.00001 \
  --author "Author A" \
  --year 2024

python3 -m cli paper validate 2405.00001
python3 -m cli paper study 2405.00001 --suite smoke
```

生成的目录：

```text
paper_reproductions/2405.00001/
├── paper.yaml
├── README.md
├── recipes/
│   ├── baseline_smoke.yaml
│   └── proposed_smoke.yaml
├── implementation/
│   └── components.py
├── tests/
│   └── README.md
└── notes/
    └── reading.md
```

官方实现或第三方仓库仍应放入 `external_projects/<project>/repo`。论文目录只保存仓库 URL/revision、适配代码、实验配方和差异记录。

## Paper Manifest

`paper.yaml` 保存论文事实和复现合同：

```yaml
schema_version: 1
paper:
  id: 2405.00001
  title: Paper Title
  authors: [Author A]
  year: 2024
  url: https://arxiv.org/abs/2405.00001
  tags: [attention, edge]

implementation:
  status: experimental
  upstream_repository: https://github.com/org/project
  upstream_revision: 0123456789abcdef

claims:
  - id: memory_reduction
    statement: Proposed attention reduces peak memory by at least 20%.
    source: Table 3, sequence length 4096
    expectations:
      - type: comparison
        recipe: proposed_smoke
        baseline: baseline_smoke
        metric: peak_memory_mb
        mode: ratio
        operator: <=
        value: 0.8

suites:
  smoke:
    recipes: [baseline_smoke, proposed_smoke]
    claims: [memory_reduction]
    strategy: sequential
```

一个 claim 可以有多个 expectation，所有 expectation 通过后 claim 才通过。没有 expectation 的 claim 会显示为 `NOT ASSESSED`，不会被伪装成成功。

## 验收规则

绝对阈值：

```yaml
- type: absolute
  recipe: proposed_full
  metric: validation_perplexity
  operator: <=
  value: 8.5
```

近似原论文数值：

```yaml
- type: absolute
  recipe: proposed_full
  metric: accuracy
  operator: approx
  value: 0.712
  tolerance: 0.005
```

相对 baseline：

```yaml
- type: comparison
  recipe: proposed_full
  baseline: baseline_full
  metric: tokens_per_second
  mode: percent_change
  operator: ">="
  value: 15
```

支持 `ratio`、`delta`、`percent_change` 三种比较模式，以及 `==`、`!=`、`<`、`<=`、`>`、`>=`、`approx` 操作符。指标名支持点路径，例如 `benchmark.prefill.tokens_per_second`。

## Recipe

Recipe 应尽量继承稳定的基础配置，只声明论文实验的差异：

```yaml
schema_version: 1
name: proposed_smoke
base_config: configs/smoke.yaml

overrides:
  experiment:
    name: paper-proposed-smoke
  extensions:
    paths:
      - paper_reproductions/2405.00001/implementation/components.py
  model:
    attention_type: paper_attention
  training:
    max_steps: 20

execution:
  executor:
    type: local
  artifact_store:
    type: local
    root: artifacts/papers
```

`extensions.paths` 可以加载单个论文实现文件并触发 registry 装饰器。论文代码只新增论文提出的部件；模型构建、训练、benchmark 和远端执行继续复用主系统。

## Smoke 与 Full Scale

建议一篇论文至少配置两类 suite：

| Suite | 目的 | 典型资源 |
| --- | --- | --- |
| `smoke` | shape、梯度、数值稳定性、流程和指标采集 | CPU 或最小 GPU |
| `ablation` | 对比单个变量，验证机制解释 | 单 GPU、小数据 |
| `full` | 接近论文规模与评测协议 | 云端 GPU、多次种子 |
| `edge` | 端侧延迟、内存、能耗和导出兼容性 | 目标设备 |

临时覆盖所有 recipe：

```bash
python3 -m cli paper study 2405.00001 \
  --suite smoke \
  --set training.max_steps=5
```

切换到云端 Executor：

```bash
python3 -m cli paper study 2405.00001 \
  --suite full \
  --executor huggingface_jobs \
  --detach
```

异步任务完成后：

```bash
python3 -m cli paper assess 2405.00001 <study-id> --wait
```

远端运行仍要求代码和 paper workspace 已提交到 Git；未提交的论文实现不会出现在云端固定 revision 中。

## 报告和结论边界

Study 状态和报告位于 `.edgellm/paper-studies/<paper-id>/<study-id>/`。报告包括 recipe/job/artifact、每条 expectation 的观测值和 PASS/FAIL，以及论文结论解释提示。

正式记录复现成功前，至少核对：数据集版本和许可证、预处理、Tokenizer、模型规模、初始化、随机种子、优化器、训练 token 数、评测脚本、精度、kernel、硬件、batch 方式和统计方差。任何差异都应写入 `notes/reading.md`，而不是只保留一个最终分数。
