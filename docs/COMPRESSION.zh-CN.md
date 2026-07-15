# 压缩系统设计

[English](COMPRESSION.md)

压缩是 EdgeLLM-Lab 的一等 Level 1/Level 2 技术域。它既要支持自己实现算法、
统一比较实验，也要能在后续把部署交给成熟端侧运行时，但不能把这三件事揉在一起。

## 分层边界

```text
张量算法
  -> 模块选择器
  -> 模型变换
  -> 压缩报告
  -> 后端/导出 adapter
  -> 目标设备 benchmark
```

- `compression/quantization/`：可读的量化算法、位打包、参考模块。
- `compression/pruning/`：mask 算法和剪枝模型变换。
- `compression/selection.py`：路径 include/exclude 和模型语义作用域。
- `compression/report.py`：统一、机器可读的压缩统计。
- `experiments/stages/`：配置驱动的流水线编排。
- `benchmark/`：量化误差和有效稀疏率测量。
- `backend/`：优化 kernel、运行时格式和真机测量。

这些边界必须保持独立。算法数值正确不等于运行更快；稠密张量中出现零值不等于
得到更小的稀疏文件；权重完成位打包也不等于所有后端都能直接执行。

## 当前算法

### 量化

| 名称 | 粒度 | 表示 | 状态 |
| --- | --- | --- | --- |
| `int8` | per-tensor 或 per-channel | 有符号 INT8 + scale | 已验证参考实现 |
| `int4` | 最后一维 group-wise | packed signed nibble + group scale | 已验证参考实现 |
| KV cache | 可配置张量量化器 | 带布局元数据的量化 K/V | 已验证参考实现 |

`ReferenceQuantizedLinear` 保存量化权重，并在可移植的 PyTorch forward 中反量化。
它用于正确性、checkpoint 和存储实验，不包含 fused low-bit GEMM，因此不会宣称延迟收益。
该模块仅用于推理；可训练低比特流程应由后续 QAT/fake-quantization 路径承担。

AWQ、GPTQ 文件目前仍是规划标记，尚未注册为可用算法。它们需要先具备校准数据、
逐层编排和与成熟实现的 parity test，才能标记为支持。

### 剪枝

| 名称 | 模式 | 主要配置 | 状态 |
| --- | --- | --- | --- |
| `magnitude` | 非结构化 | sparsity、layer/global scope | 已验证参考实现 |
| `channel` | 结构化行/列 | sparsity、axis、Lp norm | 已验证参考实现 |
| `nm` / `2:4` | 半结构化 N:M | keep、block size、axis | 已验证参考实现 |

直接 mask 会把稠密权重写成零。设置 `enforce_mask: true` 后使用 PyTorch
parametrization，使继续训练时被剪权重始终保持为零。两者仍是稠密表示；只有完成稀疏
格式导出并接入兼容后端，才能获得真实文件体积或速度收益。

## 配置方法

```yaml
compression:
  pruning:
    type: magnitude
    sparsity: 0.5
    scope: global
    enforce_mask: false
    selector:
      scopes: [language]
      include: ["*.attn.*", "*.mlp.*"]
      exclude: ["*.lm_head"]
  quantization:
    - id: language_int4
      type: int4
      group_size: 32
      selector:
        scopes: [language]
    - id: vision_int8
      type: int8
      granularity: channel
      selector:
        scopes: [vision]

pipeline:
  stages:
    - runtime_setup
    - build_model
    - prune_model
    - quantize_model
    - checkpoint
```

可直接运行的纯文本示例位于 `configs/compression/tiny_gpt_prune_int4.yaml`。
每类压缩既可以使用单个 mapping，也可以使用按顺序执行且互不重叠的 pass 列表。
selector 可以使用路径模式、语义作用域或同时使用两者。默认情况下选择结果为空会失败；
只有显式设置 `allow_empty: true` 才允许跳过。重叠剪枝 pass 会被拒绝，避免汇总报告失真。
默认拒绝 tied/shared weight；只有显式设置 `allow_shared_weights: true`，确认量化可能
打破权重共享或剪枝会影响所有 owner 后才允许执行。

## 多模态压缩

多模态模型通常需要分别处理语言模型、视觉/音频编码器、projector 和任务 head。
模型只需统一声明一次路径：

```python
def compression_scopes(self):
    return {
        "language": ("language_model",),
        "vision": ("vision_encoder",),
        "audio": ("audio_encoder",),
        "projector": ("multimodal_projector",),
}
```

`tiny_vlm` 已经实现该契约，提供 `language`、`vision`、`projector`、
`resampler`、`fusion` 和汇总 `multimodal` scope。

压缩 recipe 只面向 scope，不导入或判断具体模型类。这样可组合出“语言权重 INT4、
视觉权重 INT8、projector 保持浮点、KV cache 独立量化”等策略，并按 scope 生成报告。

共享模型输入契约支持 `input_ids`、`attention_mask`、`pixel_values`、
`audio_values` 等关键字字典。数据预处理和模态融合不属于压缩模块职责。

## 测量契约

每个压缩实验至少应记录：

- 同一评测集上的压缩前后质量；
- 数值检查的 MAE、最大绝对误差、MSE、relative MSE；
- 被选参数量和整模参数量；
- 被选权重表示字节数、整模字节数及各自压缩比；
- 分模块及全局有效稀疏率；
- 目标设备上的峰值内存、prefill/decode latency、TTFT、TPOT、tokens/s；
- backend、kernel、设备、dtype、batch size、序列长度和 warm-up 策略。

参考存储统计必须和后端运行数据分开。只有目标设备 benchmark 才能证明延迟或内存收益。

## 扩展规则

每种技术使用独立文件，并注册到对应 registry。新张量量化器实现
`TensorQuantizer`，新剪枝算法实现 `BasePruner`。算法文件不负责遍历模型。
测试至少覆盖精确 packing/mask、边界 shape、数值误差、报告、checkpoint round trip；
存在成熟实现时还应加入 parity test。

后端 adapter 使用统一报告和 model/checkpoint，但不能进入参考算法层。未来典型接入目标包括
TorchAO、bitsandbytes、AutoGPTQ/AutoAWQ、ONNX Runtime、TensorRT-LLM、MLC、
ExecuTorch、Core ML 和 llama.cpp/GGUF。

## 后续路线

1. 校准数据集、observer、激活统计和可复用 calibration cache。
2. SmoothQuant、GPTQ、AWQ、激活量化、混合精度和 QAT。
3. one-shot/渐进式剪枝、movement pruning、Wanda/SparseGPT 和恢复微调。
4. 稀疏序列化、N:M kernel adapter 和导出校验。
5. 在质量、体积、内存、延迟约束下进行逐层敏感度搜索。
6. 多模态校准采样和按 scope 的混合压缩策略。
7. 后端 parity 和目标设备 benchmark 矩阵。
