# 多模态架构

[English](MULTIMODAL.md)

多模态能力直接扩展现有组件、实验、压缩和报告体系，不建立第二套 runner。当前首个
参考实现是 `tiny_vlm`：由视觉 encoder、token resampler、projector、prefix fusion
和 decoder-only 语言模型组成的 LLaVA 类学习闭环。

## 当前目录

```text
modules/
├── vision/encoder/
│   ├── registry.py
│   └── patch_transformer.py
└── multimodal/
    ├── types.py
    ├── projector/
    │   ├── linear.py
    │   └── mlp.py
    ├── resampler/
    │   ├── identity.py
    │   └── adaptive_pool.py
    └── fusion/
        └── prefix.py

models/multimodal/
├── output.py
└── tiny_vlm.py

data/datasets/
└── synthetic_vision_language.py
```

每种技术使用独立文件和 registry。`tiny_vlm` 只组合所选实现，不拥有 projector、
resampler 或 fusion 算法。

## 可运行训练闭环

```bash
python3 -m cli train -c configs/multimodal/tiny_vlm_smoke.yaml
```

CPU smoke 执行以下路径：

```text
图像 patches
  -> 双向 patch Transformer
  -> mask-aware 固定 token resampler
  -> 非线性 projector
  -> 视觉 prefix + 文本 embedding
  -> causal Transformer
  -> 文本 logits
```

合成数据是确定性的，最后一个目标 token 由图像亮度类别决定，因此梯度必须进入视觉
encoder 和 projector；图像不是只经过接口但不参与目标的装饰输入。

## 已实现组件

| Registry | 实现 | 契约 |
| --- | --- | --- |
| `vision_encoder` | `patch_transformer` | Pixels 到 `ModalityFeatures` |
| `multimodal_projector` | `linear`、`mlp` | 模态 hidden size 到 LM hidden size |
| `multimodal_resampler` | `identity`、`adaptive_pool` | 保留或约束模态 token budget |
| `multimodal_fusion` | `prefix` | 融合模态/文本并返回文本位置 |
| `model` | `tiny_vlm` | 端到端视觉语言 causal LM |
| `dataset` | `synthetic_vision_language` | 确定性图像条件 smoke 数据 |

语言塔和视觉塔可以分别设置 block、attention、norm、MLP、hidden size、层数和 head 数。
端侧架构中，小视觉塔与语言 decoder 通常需要不同设计，因此不能共享隐式默认配置。

## Batch 契约

默认训练 Stage 接收关键字字典，不依赖纯文本签名。当前 VLM batch 为：

```python
{
    "input_ids": LongTensor[B, T],
    "attention_mask": BoolTensor[B, T],
    "pixel_values": FloatTensor[B, N, C, H, W],
    "image_mask": BoolTensor[B, N],
    "labels": LongTensor[B, T],
    "_sample_id": list[str],
}
```

`models/io.py` 递归移动嵌套 Tensor，排除 labels 和 `_` 前缀元数据，同时把完整 batch
保留给自定义 loss。其他模型族可通过 `training.model_input_keys` 限定输入字段。

## 模型输出契约

`MultimodalCausalLMOutput` 使用命名字段：

- `logits`：与 labels 对齐的文本位置 logits；
- `modality_token_count`：每个样本的有效模态 token 数；
- `text_hidden_states`：供辅助目标使用的 decoder states；
- `modality_hidden_states`：投影后的模态 states；
- `modality_attention_mask`：有效投影模态 token。

后续 alignment、contrastive、grounding、distillation loss 无需修改 trainer。trainer 已经
把 `model_outputs`、`model_inputs` 和原始 batch 传给注册 loss。

## 训练模式

`tiny_vlm` 支持三种基础模式：

- 全量端到端训练；
- `freeze_vision_encoder: true`，只适配 projector/LM；
- 同时冻结语言与视觉塔，只训练 projector。

Optimizer 参数策略会自动排除冻结参数。更复杂的分阶段 schedule 应写成显式 Stage 或
recipe，不应在通用 runner 中增加条件分支。

## 压缩 Scope

模型暴露以下稳定 scope：

```python
{
    "language": (...),
    "vision": ("vision_encoder",),
    "projector": ("multimodal_projector",),
    "resampler": ("resampler",),
    "fusion": ("fusion",),
    "multimodal": (...),
}
```

量化和剪枝 recipe 可以分别处理 language、vision 和 projector。当前模型变换面向
Linear；Conv2d 低比特变换需要单独实现并完成验证，不能复用 Linear 格式后宣称支持。

## 端侧测量

多模态报告必须分开记录：

- 图像 decode/resize/tiling 延迟；
- 视觉 encoder 延迟和峰值内存；
- resampler/projector 延迟及输出 token 数；
- 模态展开后的 LM prefill 延迟；
- decode 延迟和 KV cache 内存；
- TTFT 与端到端延迟；
- 每个 scope 压缩前后的质量。

图像分辨率、图像数量、模态 token 数、文本长度、batch size、backend、dtype 和设备
都是必填 benchmark 维度。只报告 decoder tokens/s 不足以描述端侧多模态系统。

## 后续架构阶段

1. 真实图像 processor、对话数据 adapter 和可缓存视觉特征。
2. placeholder-token replacement、可变分辨率和 multi-tile fusion。
3. 独立实现 Perceiver Resampler、Q-Former 和 gated cross-attention。
4. 视觉语言 SFT、alignment/contrastive loss 和分阶段冻结 schedule。
5. 复用视觉特征与 KV cache 的多模态生成。
6. 通过同一 feature/fusion 契约增加音频 encoder 与 projector。
7. 在后端支持时完成 llama.cpp、ONNX、MLC、ExecuTorch、Core ML 导出和真机 parity。

每个阶段都需要 CPU shape test、CUDA smoke、checkpoint round trip、可获得时的参考实现
parity test 和目标设备报告。
