# Compression Architecture

[中文说明](COMPRESSION.zh-CN.md)

Compression is a first-class Level 1 and Level 2 domain. It is designed for
learning algorithms, comparing recipes, and later delegating deployment to
mature edge runtimes without mixing those concerns.

## Boundaries

```text
tensor algorithm
  -> module selector
  -> model transform
  -> compression report
  -> backend/export adapter
  -> target-device benchmark
```

- `compression/quantization/`: readable quantizers, packing, reference modules.
- `compression/pruning/`: mask algorithms and pruning transforms.
- `compression/selection.py`: include/exclude patterns and semantic model scopes.
- `compression/report.py`: common, machine-readable accounting.
- `experiments/stages/`: config-driven orchestration.
- `benchmark/`: numerical error and effective sparsity measurements.
- `backend/`: optimized kernels, runtime formats, and device measurements.

Keeping these boundaries separate is required. An algorithm can be correct
without being fast; a sparse dense tensor is not a smaller sparse artifact; and
a packed weight format is not automatically executable by every backend.

## Current Algorithms

### Quantization

| Name | Granularity | Representation | Status |
| --- | --- | --- | --- |
| `int8` | per-tensor or per-channel | signed INT8 plus scale | verified reference |
| `int4` | groups on the last axis | packed signed nibbles plus group scales | verified reference |
| KV cache | configurable tensor quantizer | quantized K and V with layout metadata | verified reference |

`ReferenceQuantizedLinear` stores quantized weights and dequantizes during its
portable PyTorch forward pass. It is intended for correctness, checkpoint, and
storage experiments. It does not provide a fused low-bit GEMM and therefore
does not claim lower latency. It is inference-only; trainable low-bit workflows
belong to a future QAT/fake-quantization path.

AWQ and GPTQ files are still planning markers, not registered implementations.
They need calibration data, layer-wise orchestration, and parity tests before
being presented as supported.

### Pruning

| Name | Pattern | Main configuration | Status |
| --- | --- | --- | --- |
| `magnitude` | unstructured | sparsity, layer/global scope | verified reference |
| `channel` | structured rows or columns | sparsity, axis, Lp norm | verified reference |
| `nm` / `2:4` | semi-structured N:M | keep, block size, axis | verified reference |

Direct masking writes zeros into dense weights. `enforce_mask: true` installs a
PyTorch parametrization so pruned weights stay zero during further training.
Both remain dense representations; speed and file-size benefits require a
sparse export format and a compatible backend.

## Configuration

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

The runnable text-only example is
`configs/compression/tiny_gpt_prune_int4.yaml`. Each family accepts either one
mapping or an ordered list of disjoint passes. A selector can use path patterns,
semantic scopes, or both. A pass fails when nothing matches unless
`allow_empty: true` is explicitly set; overlapping pruning passes are rejected
to keep aggregate reports meaningful. Tied/shared weights are rejected by
default; `allow_shared_weights: true` is an explicit acknowledgement that a
quantization transform may break the tie or pruning may affect every owner.

## Multimodal Compression

Multimodal models often require different policies for the language model,
vision/audio encoder, projector, and task heads. A model declares paths once:

```python
def compression_scopes(self):
    return {
        "language": ("language_model",),
        "vision": ("vision_encoder",),
        "audio": ("audio_encoder",),
        "projector": ("multimodal_projector",),
    }
```

Compression recipes then target scopes without importing or checking a concrete
model class. This supports mixed policies such as INT4 language weights, INT8
vision weights, an uncompressed projector, and independently quantized KV
caches. Scope-level reports make those choices comparable.

The shared model I/O contract accepts keyword dictionaries such as
`input_ids`, `attention_mask`, `pixel_values`, and `audio_values`. Compression
does not own preprocessing or modality fusion.

## Measurement Contract

Every compression experiment should report:

- quality before and after compression on the same evaluation set;
- MAE, maximum absolute error, MSE, and relative MSE for numerical checks;
- selected and whole-model parameter counts;
- represented selected-weight bytes and whole-model bytes with separate ratios;
- effective sparsity by module and globally;
- peak memory, prefill/decode latency, TTFT, TPOT, and tokens/s on the target;
- backend, kernel, device, dtype, batch size, sequence length, and warm-up policy.

Reference storage numbers and backend runtime numbers must stay separate. Only
target-device benchmarks can support a latency or memory improvement claim.

## Extension Rules

Add one file per technique and register it through the relevant registry. A new
tensor quantizer implements `TensorQuantizer`; a pruner implements `BasePruner`.
Do not put model traversal into an algorithm file. Add tests for exact packing
or masks, edge shapes, numerical error, reports, checkpoint round trips, and
parity with a mature implementation when available.

Backend-specific adapters should consume the common report and model/checkpoint
but live outside the reference algorithm. Typical future adapters include
TorchAO, bitsandbytes, AutoGPTQ/AutoAWQ, ONNX Runtime, TensorRT-LLM, MLC, ExecuTorch,
Core ML, and llama.cpp/GGUF.

## Planned Work

1. Calibration datasets, observers, activation statistics, and reusable caches.
2. SmoothQuant, GPTQ, AWQ, activation quantization, mixed precision, and QAT.
3. One-shot and gradual pruning, movement pruning, Wanda/SparseGPT, and recovery fine-tuning.
4. Sparse serialization, N:M kernel adapters, and export validation.
5. Layer sensitivity search under quality, size, memory, and latency constraints.
6. Multimodal calibration sampling and per-scope mixed compression policies.
7. Backend parity and target-device benchmark matrices.
