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
├── core/                    # 注册表、组件规范、配置、环境追踪、运行时工具
├── modules/                 # Level 1: 算法包；每种技术独立文件
├── models/                  # Level 1: TinyGPT、LLaMA-like、DeepSeek-like、后续模型族
├── training/                # Level 1/2: Loss、Optimizer、Scheduler、训练入口
├── inference/               # Level 1/2: Sampler、KV Cache、生成引擎
├── compression/             # Level 1/2: 量化、剪枝、低秩压缩
├── data/                    # Tokenizer、Dataset、Dataloader
├── experiments/             # Level 2: 可注册 Stage 流水线、上下文和产物管理
├── benchmark/               # Level 2: Benchmark 注册表和指标采集
├── backend/                 # Level 3: Torch、llama.cpp、ONNX、MLC 等运行时边界
├── integrations/            # Level 3: 外部开源项目的轻量适配层
├── external_projects/       # 外部项目源码独立工作区，不属于核心包
├── configs/                 # 实验配置
├── docs/                    # 架构和开源项目接入文档
├── tests/                   # 单元测试和手动调试探针
├── reports/                 # 学习笔记和实验报告
├── deploy/                  # 端侧部署实验
├── cli.py                   # CLI 入口
└── main.py                  # 兼容 CLI 入口
```

## Level 1: Learn by Implementing

这一层用于自己实现 LLM 核心部件，通过代码理解原理。

`modules/` 采用“技术域一个包、每种技术一个文件”的结构：

```text
modules/
├── attention/
│   ├── mha.py
│   ├── mqa.py
│   ├── gqa.py
│   ├── mla.py
│   ├── sliding_window.py
│   └── sparse.py
├── mlp/
│   ├── gelu.py
│   └── swiglu.py
├── norm/
│   ├── layernorm.py
│   └── rmsnorm.py
├── block/
│   └── transformer.py
├── position/
│   └── rope.py
└── moe/
    └── router.py
```

`tests/` 与可直接使用的 `modules/` 分离：

```text
tests/
├── unit/     # 稳定单元测试和 smoke test
└── debug/    # 手动运行的调试探针
```

当前已经支持可替换的部件：

- `attention`: 当前有 MHA、MQA、GQA、学习版 MLA、滑动窗口、top-k 稀疏 Attention。
- `mlp`: GELU MLP 和 SwiGLU。
- `norm`: LayerNorm 和 RMSNorm。
- `block`: Transformer Block。
- `position_encoding`: RoPE。
- `model`: 当前有 TinyGPT，后续扩展 LLaMA-like、SmolLM-like、MobileLLM-like。
- `loss`: causal LM cross entropy、z-loss、distillation。
- `sampler`: greedy、multinomial、top-k、top-p。
- `kv_cache`: 当前有 append-only cache，后续扩展 paged、sliding、quantized cache。
- `quantizer`: 当前只有实验性的张量级 INT8；packed INT4、AWQ、GPTQ 和 KV cache quantization 仍在规划中。

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

注册项包含 Level、成熟度、能力、依赖、别名和源码位置等结构化信息，且无需实例化模型即可查看：

```bash
python3 -m cli list-components --details
python3 -m cli component-info backend llama_cpp
```

## Level 2: Experiment by Comparing

这一层用于在统一框架下比较不同算法、模型结构、压缩方法和推理后端。

实验流程：

```text
config.yaml
  -> 可注册的 Pipeline Stages
  -> build / train / evaluate / compress / export / benchmark
  -> 采集指标和运行环境
  -> 写入实验产物
```

默认 Pipeline 保持原有训练行为。扩展模块可以注册新的 `ExperimentStage`，然后在 `pipeline.stages` 中排列，无需修改 `ExperimentRunner`。

每次运行都会生成独立目录：

```text
runs/<run-id>/
├── config.yaml
├── metrics.json
├── manifest.json
└── report.md
```

`manifest.json` 记录运行状态、环境、Git revision、选用组件、Stage 耗时、错误和产物路径。

快速 smoke test：

```bash
python3 -m cli smoke
```

按配置运行实验：

```bash
python3 -m cli train -c configs/smoke.yaml
```

不执行实验，仅校验并展开配置：

```bash
python3 -m cli validate-config -c configs/smoke.yaml
python3 -m cli validate-config -c configs/smoke.yaml --resolved
```

查看已注册组件：

```bash
python3 -m cli list-components
```

运行稳定测试：

```bash
python3 -m unittest discover -s tests
```

运行手动调试探针：

```bash
python3 -m tests.debug.attention_variants_debug
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
- [扩展开发指南](docs/EXTENDING.md)
- [开源项目接入流程](docs/OPEN_SOURCE_INTEGRATION.md)
