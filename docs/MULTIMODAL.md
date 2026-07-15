# Multimodal Architecture Plan

[中文说明](MULTIMODAL.zh-CN.md)

Multimodal support should extend the existing component system instead of
creating a second training framework. The first implementation target is a
LLaVA-style path: a pretrained modality encoder, a trainable projector, and a
decoder-only language model. Cross-attention and native multimodal token models
can be added after that end-to-end path is reproducible.

## Stable Contract Available Now

The default training stage no longer assumes that a batch only contains
`input_ids`. `models/io.py`:

- recursively moves nested inputs to the selected device;
- separates labels and `_`-prefixed metadata;
- calls models with keyword arguments;
- extracts logits from tensors, mappings, or object-style outputs.

A future batch can therefore use this shape without changing the runner:

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

Only fields consumed by the selected model should be configured through
`training.model_input_keys`. Metadata remains available to losses and metrics
through the original batch.

## Planned Component Boundaries

```text
data processors
  -> modality encoder
  -> projector / resampler
  -> fusion strategy
  -> language model
  -> task loss and generation policy
```

- Processor: decode, resize/tile, normalize, tokenize, and align placeholders.
- Encoder: vision, audio, or another modality tower.
- Projector/resampler: map variable modality features to the LM hidden space and token budget.
- Fusion: token insertion, cross-attention, gated cross-attention, or native mixed tokens.
- Model family: own the composition and expose stable semantic scopes.
- Loss: language modeling plus optional contrastive, alignment, grounding, or auxiliary objectives.
- Inference: own modality feature caching, prompt expansion, and generation state.

Each technique belongs in an independent file and registry. Project-specific
wrappers for mature encoders stay in `integrations/`; internal learning
implementations stay in `modules/` and `models/`.

## Model Contract

A multimodal model should accept keyword inputs and return either a logits
tensor or an output carrying `logits`. Optional intermediate values should use
named fields, not positional tuples. It should also define compression scopes:

```python
def compression_scopes(self):
    return {
        "language": ("language_model",),
        "vision": ("vision_encoder",),
        "projector": ("multimodal_projector",),
    }
```

The composition owns special-token expansion and modality positions. The
generic runner must not know how image patches or audio frames are inserted.

## Edge-Side Measurement

Multimodal reports must separate:

- processor latency;
- encoder latency and peak memory;
- projector/resampler latency and output token count;
- LM prefill latency after modality expansion;
- decode latency and KV cache memory;
- time to first token and end-to-end latency;
- quality before and after per-scope compression.

Image resolution, tile count, audio duration, modality token count, text length,
batch size, backend, and device are mandatory benchmark dimensions. Reporting
only decoder tokens/s is insufficient for an edge multimodal system.

## Compression Rules

Calibration data must exercise every enabled modality and realistic modality
token lengths. Quantization and pruning recipes target semantic scopes, so a
single model can use different bit widths and sparsity patterns per tower.
Projectors and normalization layers should remain selectable exclusions because
they can be more sensitive than large matrix layers.

Weight representation, activation memory, modality feature cache, and KV cache
must be measured independently. A backend adapter must declare which scopes,
operators, dynamic shapes, and cache formats it can execute.

## Delivery Phases

1. Contract tests with a tiny synthetic vision-language model and dataset.
2. LLaVA-style encoder/projector/decoder training and generation smoke path.
3. Frozen-encoder projector training, full SFT, and modality feature caching.
4. Per-scope INT8/INT4 and pruning comparison with quality regression gates.
5. Cross-attention/resampler architectures and multi-image variable-resolution input.
6. Audio and additional modalities through the same batch and scope contracts.
7. llama.cpp, ONNX, MLC, ExecuTorch, or Core ML export where capability permits.

Every phase needs CPU shape tests, CUDA smoke tests, checkpoint round trips,
reference parity where possible, and target-device benchmark reports.
