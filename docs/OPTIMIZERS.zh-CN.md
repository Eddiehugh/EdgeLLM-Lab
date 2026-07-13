# 优化器架构

优化器按独立扩展轴拆分，避免算法公式、参数选择和高性能后端互相耦合。

```text
training/optimizers/
├── api.py                 配置解析与统一构建入口
├── registry.py            实现注册表和参数策略注册表
├── reference/             Level 1 可读参考实现
├── adapters/              Level 3 成熟库适配器
└── policies/              模型参数分组规则
```

当前内置能力：

- `reference_adamw`：用于学习和数值对齐测试的可读 AdamW。
- `torch_adamw` / `adamw`：工作使用的 PyTorch AdamW。
- `torch_sgd` / `sgd`：工作使用的 PyTorch SGD。
- `all`：所有可训练参数使用一个参数组。
- `decay_by_dimension`：矩阵参数应用 weight decay，向量和标量不应用。

## 配置方式

新实验使用结构化配置：

```yaml
training:
  optimizer:
    algorithm: adamw
    implementation: torch
    param_group_policy:
      type: decay_by_dimension
      min_decay_ndim: 2
    lr: 0.0003
    betas: [0.9, 0.95]
    weight_decay: 0.1
```

`algorithm` 和 `implementation` 会解析成
`<implementation>_<algorithm>`。旧配置 `{type: adamw}` 继续兼容，并映射到
`torch_adamw`。

除 `all` 之外的参数策略要求构建时传入 `model=`，因为
`model.parameters()` 会丢失参数名称和模型结构信息。

## 扩展规则

研究更新公式时，在 `reference/` 中增加可读实现；复用 PyTorch、bitsandbytes、
DeepSpeed 等稳定包依赖时，在 `adapters/` 中增加薄适配器。nanochat 这类外部源码
仍放在 `external_projects/`，由 `integrations/` 或扩展机制注册优化器，核心训练包
不能直接依赖外部源码目录。不同实现应在 `tests/parity/optimizers/` 中进行数值对齐。

参数路由规则独立放在 `policies/`。例如“矩阵使用 Muon，Embedding 和标量使用
AdamW”应由组合策略表达，不应把参数名称判断写进 Muon 的更新公式。
