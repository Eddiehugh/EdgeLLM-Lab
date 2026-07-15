# Multimodal Architecture

[中文说明](MULTIMODAL.zh-CN.md)

Multimodal support extends the normal component, experiment, compression, and
reporting systems. It does not introduce a second runner. The first implemented
reference is `tiny_vlm`, a LLaVA-style learning path with a vision encoder,
token resampler, projector, prefix fusion, and decoder-only language model.

## Current Layout

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

Every technique has an independent file and registry entry. `tiny_vlm` only
composes selected implementations; it does not own projector, resampling, or
fusion algorithms.

## Runnable Training Path

```bash
python3 -m cli train -c configs/multimodal/tiny_vlm_smoke.yaml
```

The smoke recipe trains this path on CPU:

```text
image patches
  -> bidirectional patch Transformer
  -> mask-aware fixed token resampler
  -> nonlinear projector
  -> visual prefix + text embeddings
  -> causal Transformer
  -> text logits
```

The synthetic dataset is deterministic. Its final target token is derived from
image intensity, so gradients must reach the vision encoder and projector; the
image is not an unused decorative input.

## Implemented Components

| Registry | Implementations | Contract |
| --- | --- | --- |
| `vision_encoder` | `patch_transformer` | Pixels to `ModalityFeatures` |
| `multimodal_projector` | `linear`, `mlp` | Modality hidden size to LM hidden size |
| `multimodal_resampler` | `identity`, `adaptive_pool` | Preserve or constrain modality token budget |
| `multimodal_fusion` | `prefix` | Fuse modality and text while returning text positions |
| `model` | `tiny_vlm` | End-to-end vision-language causal LM |
| `dataset` | `synthetic_vision_language` | Deterministic image-conditioned smoke data |

The language and vision towers have independent block, attention, norm, MLP,
hidden-size, layer-count, and head-count settings. This is necessary for mobile
architecture experiments where a small vision tower and language decoder often
need different design choices.

## Batch Contract

The default training stage accepts keyword dictionaries and does not depend on
a text-only signature. The current VLM batch is:

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

`models/io.py` recursively moves nested tensors, excludes labels and
underscore-prefixed metadata from model arguments, and leaves the complete batch
available to custom losses. `training.model_input_keys` can restrict inputs for
another model family.

## Model Output Contract

`MultimodalCausalLMOutput` exposes named fields:

- `logits`: text-position logits aligned with `labels`;
- `modality_token_count`: valid modality tokens per sample;
- `text_hidden_states`: decoder states for auxiliary objectives;
- `modality_hidden_states`: projected modality states;
- `modality_attention_mask`: valid projected modality tokens.

This supports future alignment, contrastive, grounding, or distillation losses
without changing the trainer. The trainer already passes `model_outputs`,
`model_inputs`, and the original batch to a registered loss.

## Training Modes

`tiny_vlm` supports three useful starting modes:

- full end-to-end training;
- `freeze_vision_encoder: true` for projector/LM adaptation;
- `freeze_language_model: true` together with a frozen vision encoder for projector-only training.

The optimizer parameter policy automatically excludes frozen parameters. More
complex staged schedules should become explicit experiment stages or recipes,
not conditionals in the generic runner.

## Compression Scopes

The model exposes stable scopes:

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

Quantization and pruning recipes can therefore target language, vision, and
projector paths independently. Current model transforms operate on Linear
layers; Conv2d low-bit transforms require a separate verified implementation.

## Edge-Side Measurement

Multimodal reports must separate:

- image decode/resize/tiling latency;
- vision encoder latency and peak memory;
- resampler/projector latency and resulting token count;
- LM prefill latency after modality expansion;
- decode latency and KV cache memory;
- TTFT and end-to-end latency;
- quality before and after compression for each scope.

Image resolution, image count, modality token count, text length, batch size,
backend, dtype, and device are mandatory benchmark dimensions. Decoder tokens/s
alone is not an adequate edge multimodal metric.

## Next Architecture Phases

1. Real image processor and conversation dataset adapters with cached visual features.
2. Placeholder-token replacement and variable-resolution/multi-tile fusion.
3. Perceiver Resampler, Q-Former, and gated cross-attention as independent techniques.
4. Vision-language SFT, alignment/contrastive losses, and staged freeze schedules.
5. Multimodal generation with reusable vision features and KV cache.
6. Audio encoders and projectors through the same feature/fusion contracts.
7. Backend export and device parity for llama.cpp, ONNX, MLC, ExecuTorch, and Core ML where supported.

Each phase requires CPU shape tests, CUDA smoke tests, checkpoint round trips,
reference parity where available, and target-device reports.
