# 多模态架构规划

[English](MULTIMODAL.md)

多模态能力应扩展现有组件体系，而不是再造一套训练框架。第一条实现路线选择
LLaVA 类结构：预训练模态 encoder、可训练 projector、decoder-only 语言模型。
这条端到端路径可复现之后，再扩展 cross-attention 和原生混合模态 token 模型。

## 当前已经具备的稳定契约

默认训练 Stage 不再假定 batch 只有 `input_ids`。`models/io.py` 已经支持：

- 递归把嵌套输入移动到目标设备；
- 分离 labels 和以 `_` 开头的元数据；
- 使用关键字参数调用模型；
- 从 Tensor、mapping 或对象式输出中提取 logits。

因此后续 batch 可以直接使用如下结构，无需修改 runner：

```python
{
    "input_ids": LongTensor[B, T],
    "attention_mask": Tensor[B, T],
    "pixel_values": Tensor[B, N, C, H, W],
    "image_sizes": Tensor[B, N, 2],
    "modality_positions": LongTensor[B, N],
    "labels": LongTensor[B, T],
    "_sample_id": list[str],
}
```

通过 `training.model_input_keys` 只选择当前模型真正消费的字段。原始 batch 中的
元数据仍会传给 loss 和 metric 使用。

## 规划中的组件边界

```text
数据 processor
  -> 模态 encoder
  -> projector / resampler
  -> 融合策略
  -> 语言模型
  -> 任务 loss 和生成策略
```

- Processor：解码、resize/tile、归一化、tokenize 和占位符对齐。
- Encoder：视觉、音频或其他模态塔。
- Projector/resampler：映射到 LM hidden space，并控制可变模态 token 数量。
- Fusion：token 插入、cross-attention、gated cross-attention 或原生混合 token。
- Model family：负责组合上述部件并暴露稳定语义 scope。
- Loss：语言建模，以及可选的对比、对齐、grounding 或辅助目标。
- Inference：负责模态特征 cache、prompt 展开和生成状态。

每种技术保持独立文件和 registry。成熟 encoder 的项目适配放在 `integrations/`；
用于学习的内部实现放在 `modules/` 和 `models/`。

## 模型契约

多模态模型应接受关键字输入，并返回 logits Tensor 或带 `logits` 的输出。可选中间量
使用命名字段，不使用位置 tuple。模型还应声明压缩作用域：

```python
def compression_scopes(self):
    return {
        "language": ("language_model",),
        "vision": ("vision_encoder",),
        "projector": ("multimodal_projector",),
    }
```

具体模型负责特殊 token 展开和模态位置。通用 runner 不应知道图像 patch 或音频 frame
如何插入语言序列。

## 端侧测量

多模态报告必须分开记录：

- processor 延迟；
- encoder 延迟和峰值内存；
- projector/resampler 延迟与输出 token 数；
- 模态展开后的 LM prefill 延迟；
- decode 延迟与 KV cache 内存；
- TTFT 和端到端延迟；
- 各 scope 压缩前后的质量。

图像分辨率、tile 数、音频时长、模态 token 数、文本长度、batch size、backend 和设备
都是必填 benchmark 维度。只报告 decoder tokens/s 无法描述端侧多模态系统。

## 压缩规则

校准数据必须覆盖每种已启用模态和真实的模态 token 长度。量化和剪枝 recipe 面向语义
scope，因此同一模型的不同塔可以使用不同 bit width 和稀疏模式。projector 与 norm
往往比大矩阵层敏感，必须能通过 selector 独立排除。

权重表示、activation 内存、模态 feature cache 和 KV cache 必须分别测量。backend
adapter 必须声明可以执行的 scope、operator、动态 shape 和 cache 格式。

## 交付阶段

1. 使用微型合成视觉语言模型和数据集完成契约测试。
2. 跑通 LLaVA 类 encoder/projector/decoder 的训练与生成 smoke。
3. 支持冻结 encoder 的 projector 训练、全量 SFT 和模态 feature cache。
4. 按 scope 对比 INT8/INT4 与剪枝，并设置质量回归阈值。
5. 扩展 cross-attention/resampler、多图和可变分辨率输入。
6. 通过同一 batch/scope 契约扩展音频等更多模态。
7. 在能力允许时接入 llama.cpp、ONNX、MLC、ExecuTorch 或 Core ML 导出。

每个阶段都需要 CPU shape test、CUDA smoke、checkpoint round trip、可行时的参考实现
parity test，以及目标设备 benchmark 报告。
