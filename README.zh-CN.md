# EdgeLLM-Lab

[English README](README.md)

EdgeLLM-Lab 是一个面向大模型算法开发工程师的个人 LLM 学习、实验和工具化平台，重点服务于小模型、端侧 LLM、模型结构实验、量化压缩和推理部署。

项目按照三个 Level 设计：

```text
Level 1: Learn by Implementing
  自己实现核心算法，用于理解原理。

Level 2: Experiment by Comparing
  在统一框架下对不同技术算法做实验和 benchmark。

Level 3: Work by Wrapping
  封装成熟开源项目能力，形成工作中可复用的工具系统。
```

项目的核心原则：

```text
学习一个技术 -> 实现一个模块 -> 跑一个实验 -> 写一份报告
```

## 系统设计

```text
EdgeLLM-Lab/
├── core/                    # 注册表、配置加载、扩展加载、运行时工具
├── modules/                 # Level 1: Attention、MLP、Norm、RoPE、Block、MoE
├── models/                  # Level 1: TinyGPT、LLaMA-like、DeepSeek-like、后续模型族
├── training/                # Level 1/2: Loss、Optimizer、Scheduler、训练入口
├── inference/               # Level 1/2: Sampler、KV Cache、生成引擎
├── compression/             # Level 1/2: 量化、剪枝、低秩压缩
├── data/                    # Tokenizer、Dataset、Dataloader
├── experiments/             # Level 2: 配置驱动实验运行器和实验产物管理
├── benchmark/               # Level 2: Benchmark 注册表和指标采集
├── backend/                 # Level 3: Torch、llama.cpp、ONNX、MLC 等运行时边界
├── integrations/            # Level 3: 外部开源项目的轻量适配层
├── external_projects/       # 外部项目源码独立工作区，不属于核心包
├── configs/                 # 实验配置
├── docs/                    # 架构和开源项目接入文档
├── reports/                 # 学习笔记和实验报告
├── deploy/                  # 端侧部署实验
├── cli.py                   # CLI 入口
└── main.py                  # 兼容 CLI 入口
```

## Level 1: Learn by Implementing

这一层用于自己实现 LLM 核心部件，通过代码理解原理。

当前已经支持可替换的部件：

- `attention`: 当前有 MHA，后续扩展 MQA、GQA、MLA、滑动窗口、稀疏 Attention。
- `mlp`: GELU MLP 和 SwiGLU。
- `norm`: LayerNorm 和 RMSNorm。
- `block`: Transformer Block。
- `position_encoding`: RoPE。
- `model`: 当前有 TinyGPT，后续扩展 LLaMA-like、SmolLM-like、MobileLLM-like。
- `loss`: causal LM cross entropy、z-loss、distillation。
- `sampler`: greedy、multinomial、top-k、top-p。
- `kv_cache`: 当前有 append-only cache，后续扩展 paged、sliding、quantized cache。
- `quantizer`: 当前有 INT8，后续扩展 INT4、AWQ、GPTQ、KV cache quantization。

例如新增自己的 Attention：

```python
from modules.attention import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("my_attention")
class MyAttention:
    ...
```

在配置中启用：

```yaml
model:
  attention_type: my_attention
```

## Level 2: Experiment by Comparing

这一层用于在统一框架下比较不同算法、模型结构、压缩方法和推理后端。

实验流程：

```text
config.yaml
  -> 构建 tokenizer / dataset / model / loss / optimizer / scheduler
  -> 训练或评估
  -> 采集指标
  -> 写入实验产物
```

每次运行都会生成独立目录：

```text
runs/<run-id>/
├── config.yaml
├── metrics.json
└── report.md
```

快速 smoke test：

```bash
python3 -m cli smoke
```

按配置运行实验：

```bash
python3 -m cli train -c configs/smoke.yaml
```

查看已注册组件：

```bash
python3 -m cli list-components
```

Level 2 重点指标：

- training loss 和 perplexity
- model size
- parameter count
- prefill latency
- decode latency
- TTFT 和 TPOT
- tokens/s
- peak memory
- KV cache memory
- quantization error
- backend runtime latency

## Level 3: Work by Wrapping

这一层用于封装成熟开源项目能力，但不把外部项目源码揉进本系统核心代码。

边界规则：

```text
external_projects/<project>/repo
  外部开源项目源码和大型产物。
  不作为本系统 Python 包导入。

integrations/<project>/
  本系统写的轻量 adapter、元信息、配置映射、转换胶水。

backend/
  当外部项目提供推理能力时，通过 backend 形成运行时边界。
```

已规划的开源项目接入目标：

| 项目 | 最快用途 | 长期角色 |
| --- | --- | --- |
| nanoGPT | 最小训练/生成闭环参考 | 吸收简单闭环思想 |
| TinyLlama | LLaMA-like 结构参考 | 映射配置和 checkpoint 结构 |
| SmolLM | 小模型族 baseline | 比较小模型规模和训练配方 |
| MobileLLM | 端侧架构设计参考 | 吸收 mobile-oriented 结构思想 |
| llama.cpp | 成熟端侧推理运行时 | GGUF、量化推理、backend benchmark |

查看外部项目适配层：

```bash
python3 -m cli list-integrations
python3 -m cli integration-info llama_cpp
python3 -m cli integration-info nanogpt --local-path /path/to/nanoGPT
```

## 外部项目管理规则

开源项目不应该 vendor 到内部框架中。

推荐放置方式：

```text
external_projects/
├── nanogpt/repo/
├── tinyllama/repo/
├── smollm/repo/
├── mobilellm/repo/
└── llama_cpp/repo/
```

不要从以下目录直接 import 外部项目源码：

- `core/`
- `experiments/`
- `modules/`
- `models/`
- `training/`

adapter 的职责是把外部项目能力翻译成本系统的 registry、config、checkpoint、benchmark target 或 backend 调用。

## 开发路线

```text
v0.1: TinyGPT + MHA + train + generate
v0.2: RoPE + RMSNorm + SwiGLU
v0.3: KV cache + streaming generation
v0.4: MQA / GQA
v0.5: INT8 / INT4 QuantLinear
v0.6: MLA
v0.7: Sliding Window / Sparse Attention
v0.8: Benchmark suite
v0.9: llama.cpp / ONNX backend
v1.0: edge deployment demo
```

## 验证命令

```bash
python3 -m compileall core modules models training inference backend compression data experiments benchmark integrations cli.py main.py
python3 -m unittest discover -s tests
python3 -m cli list-components
python3 -m cli list-integrations
python3 -m cli train -c configs/smoke.yaml
```

## 设计文档

- [架构设计](docs/ARCHITECTURE.md)
- [开源项目接入流程](docs/OPEN_SOURCE_INTEGRATION.md)
