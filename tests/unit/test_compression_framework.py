"""Behavioral tests for quantization, pruning, and multimodal-ready selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from benchmark import build_benchmark
from compression import (
    ModuleSelector,
    build_pruner,
    build_quantizer,
    prune_linear_modules,
    quantize_linear_modules,
)
from compression.pruning.magnitude import MagnitudePruner
from compression.pruning.nm import NMPruner
from compression.pruning.structured import StructuredChannelPruner
from compression.quantization.int4 import GroupwiseInt4Quantizer
from compression.quantization.int8 import SymmetricInt8Quantizer
from compression.quantization.kv_quant import KVCacheQuantizer
from compression.quantization.packing import pack_int4, unpack_int4
from compression.quantization.quant_linear import ReferenceQuantizedLinear
from experiments import run_experiment
from models import extract_logits, prepare_model_inputs


class _ScopedMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.vision_encoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.projector = nn.Linear(8, 8)

    @staticmethod
    def compression_scopes():
        return {
            "language": ("language_model",),
            "vision": ("vision_encoder",),
            "projector": ("projector",),
        }


class QuantizationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_int8_per_channel_round_trip_and_storage(self) -> None:
        tensor = torch.randn(8, 16)
        quantizer = SymmetricInt8Quantizer(granularity="channel", axis=0)

        quantized = quantizer.quantize(tensor)
        reconstructed = quantizer.dequantize(quantized)

        self.assertEqual(quantized.scale.shape, (8,))
        self.assertEqual(reconstructed.shape, tensor.shape)
        self.assertLess(quantized.storage_bytes, tensor.numel() * tensor.element_size())
        self.assertLess(float((tensor - reconstructed).abs().max()), 0.03)

    def test_int4_packing_is_exact_for_odd_value_count(self) -> None:
        values = torch.tensor([-8, -7, -1, 0, 1, 6, 7], dtype=torch.int8)

        packed = pack_int4(values)
        unpacked = unpack_int4(packed, values.numel())

        self.assertTrue(torch.equal(values, unpacked))
        self.assertEqual(packed.numel(), 4)

    def test_groupwise_int4_round_trip_uses_packed_storage(self) -> None:
        tensor = torch.randn(3, 10)
        quantizer = GroupwiseInt4Quantizer(group_size=4)

        quantized = quantizer.quantize(tensor)
        reconstructed = quantizer.dequantize(quantized)

        self.assertTrue(quantized.packed)
        self.assertEqual(quantized.values.dtype, torch.uint8)
        self.assertEqual(reconstructed.shape, tensor.shape)
        self.assertLess(quantized.storage_bytes, tensor.numel() * tensor.element_size())
        self.assertLess(float((tensor - reconstructed).abs().mean()), 0.15)

    def test_quantizers_reject_non_finite_and_non_floating_inputs(self) -> None:
        with self.assertRaises(TypeError):
            SymmetricInt8Quantizer().quantize(torch.ones(4, dtype=torch.int64))
        with self.assertRaises(ValueError):
            GroupwiseInt4Quantizer().quantize(torch.tensor([[float("nan")]]))

    def test_quantized_linear_replaces_weights_without_claiming_fast_kernel(self) -> None:
        linear = nn.Linear(16, 8)
        inputs = torch.randn(4, 16)
        reference = linear(inputs)

        quantized = ReferenceQuantizedLinear.from_float(
            linear, SymmetricInt8Quantizer(granularity="channel", axis=0)
        )
        output = quantized(inputs)

        self.assertLess(quantized.weight_storage_bytes, linear.weight.numel() * 4)
        self.assertTrue(torch.allclose(reference, output, atol=0.03, rtol=0.03))
        self.assertNotIn("weight", dict(quantized.named_parameters()))

    def test_model_quantization_respects_multimodal_scope(self) -> None:
        model = _ScopedMultimodalModel()
        selector = ModuleSelector(scopes=("vision",))

        transformed, report = quantize_linear_modules(
            model,
            SymmetricInt8Quantizer(),
            selector=selector,
        )

        self.assertIsInstance(transformed.vision_encoder[0], ReferenceQuantizedLinear)
        self.assertIsInstance(transformed.language_model[0], nn.Linear)
        self.assertIsInstance(transformed.projector, nn.Linear)
        self.assertEqual([record.name for record in report.records], ["vision_encoder.0"])
        self.assertFalse(report.metadata["latency_accelerated"])

    def test_quantized_model_state_dict_round_trip(self) -> None:
        source = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        source, _ = quantize_linear_modules(
            source,
            GroupwiseInt4Quantizer(group_size=4),
        )
        inputs = torch.randn(3, 8)

        target = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        target, _ = quantize_linear_modules(
            target,
            GroupwiseInt4Quantizer(group_size=4),
        )
        target.load_state_dict(source.state_dict())

        self.assertTrue(torch.equal(source(inputs), target(inputs)))

    def test_tied_weight_transform_requires_explicit_acknowledgement(self) -> None:
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(8, 4)
                self.head = nn.Linear(4, 8, bias=False)
                self.head.weight = self.embedding.weight

        selector = ModuleSelector(include=("head",))
        with self.assertRaisesRegex(ValueError, "tied weight"):
            quantize_linear_modules(
                TiedModel(),
                SymmetricInt8Quantizer(),
                selector=selector,
            )

        transformed, report = quantize_linear_modules(
            TiedModel(),
            SymmetricInt8Quantizer(),
            selector=selector,
            allow_shared_weights=True,
        )
        self.assertIsInstance(transformed.head, ReferenceQuantizedLinear)
        self.assertEqual(
            report.records[0].metadata["shared_weight_owners"],
            ["embedding.weight", "head.weight"],
        )

    def test_kv_cache_quantization_is_layout_agnostic(self) -> None:
        key = torch.randn(2, 4, 8, 16)
        value = torch.randn_like(key)
        quantizer = KVCacheQuantizer()

        cache = quantizer.quantize(key, value)
        restored_key, restored_value = quantizer.dequantize(cache)

        self.assertEqual(restored_key.shape, key.shape)
        self.assertEqual(restored_value.shape, value.shape)
        self.assertLess(cache.storage_bytes, (key.numel() + value.numel()) * 4)


class PruningTest(unittest.TestCase):
    def test_layer_and_global_magnitude_pruning_are_exact(self) -> None:
        weight = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        local_mask = MagnitudePruner(sparsity=0.5).compute_mask(weight)
        self.assertEqual(int((local_mask == 0).sum()), 8)

        weights = {
            "first": torch.arange(1, 5, dtype=torch.float32),
            "second": torch.arange(5, 9, dtype=torch.float32),
        }
        masks = MagnitudePruner(sparsity=0.25, scope="global").compute_masks(weights)
        self.assertEqual(sum(int((mask == 0).sum()) for mask in masks.values()), 2)
        self.assertFalse(bool(masks["first"][0]))
        self.assertFalse(bool(masks["first"][1]))

    def test_structured_channel_pruning_zeros_complete_rows(self) -> None:
        weight = torch.tensor(
            [[0.1, 0.1], [3.0, 3.0], [0.2, 0.2], [4.0, 4.0]]
        )
        mask = StructuredChannelPruner(sparsity=0.5, axis=0).compute_mask(weight)

        row_keep_counts = mask.sum(dim=1).tolist()
        self.assertEqual(row_keep_counts, [0, 2, 0, 2])

    def test_nm_pruning_keeps_exactly_two_of_every_four(self) -> None:
        weight = torch.randn(3, 12)
        mask = NMPruner(keep=2, block_size=4).compute_mask(weight)

        grouped = mask.reshape(3, 3, 4)
        self.assertTrue(torch.equal(grouped.sum(dim=-1), torch.full((3, 3), 2)))

    def test_dense_pruning_report_does_not_claim_storage_reduction(self) -> None:
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        transformed, report = prune_linear_modules(
            model, MagnitudePruner(sparsity=0.5, scope="global")
        )

        zero_count = sum(
            int((module.weight == 0).sum())
            for module in transformed.modules()
            if isinstance(module, nn.Linear)
        )
        total = sum(record.parameter_count for record in report.records)
        self.assertEqual(zero_count, int(total * 0.5))
        self.assertEqual(report.compression_ratio, 1.0)
        self.assertEqual(report.model_compression_ratio, 1.0)
        self.assertFalse(report.metadata["dense_storage_reduced"])

    def test_enforced_mask_survives_weight_updates(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        transformed, _ = prune_linear_modules(
            model,
            MagnitudePruner(sparsity=0.5),
            enforce_mask=True,
        )
        layer = transformed[0]
        before_mask = layer.parametrizations.weight[0].mask.bool().clone()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        optimizer.zero_grad()
        layer(torch.ones(2, 4)).sum().backward()
        optimizer.step()

        self.assertTrue(torch.equal(layer.weight[~before_mask], torch.zeros_like(layer.weight[~before_mask])))


class CompressionPipelineAndIOTest(unittest.TestCase):
    def test_compression_components_are_registered(self) -> None:
        self.assertEqual(type(build_quantizer("int4")).__name__, "GroupwiseInt4Quantizer")
        self.assertEqual(type(build_pruner("2:4")).__name__, "NMPruner")

    def test_compression_benchmarks_report_error_and_effective_sparsity(self) -> None:
        reference = torch.tensor([1.0, 2.0])
        reconstructed = torch.tensor([1.0, 2.25])
        error = build_benchmark("quantization_error")(reference, reconstructed)
        self.assertAlmostEqual(error["max_abs_error"], 0.25)

        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[0.0, 1.0], [0.0, 2.0]]))
        sparsity = build_benchmark("model_sparsity")(model)
        self.assertEqual(sparsity["zero_count"], 2)
        self.assertEqual(sparsity["sparsity"], 0.5)

    def test_pipeline_prunes_then_quantizes_and_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(
                {
                    "experiment": {
                        "name": "compression-smoke",
                        "output_dir": temporary,
                    },
                    "runtime": {"device": "cpu", "seed": 11},
                    "model": {
                        "type": "tiny_gpt",
                        "vocab_size": 64,
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 4,
                        "max_position_embeddings": 8,
                    },
                    "training": {"save_checkpoint": True},
                    "compression": {
                        "pruning": {
                            "type": "magnitude",
                            "sparsity": 0.25,
                            "scope": "global",
                        },
                        "quantization": {
                            "type": "int4",
                            "group_size": 8,
                        },
                    },
                    "pipeline": {
                        "stages": [
                            "runtime_setup",
                            "build_model",
                            "model_stats",
                            "prune_model",
                            "quantize_model",
                            "checkpoint",
                        ]
                    },
                }
            )

            self.assertGreater(result["metrics"]["pruned_module_count"], 0)
            self.assertGreater(result["metrics"]["quantized_module_count"], 0)
            self.assertGreater(result["metrics"]["quantized_weight_compression_ratio"], 1.0)
            self.assertGreater(result["metrics"]["quantized_model_compression_ratio"], 1.0)
            self.assertTrue(Path(result["metrics"]["checkpoint_path"]).exists())

    def test_pipeline_supports_disjoint_mixed_quantization_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(
                {
                    "experiment": {"name": "mixed-quant", "output_dir": temporary},
                    "runtime": {"device": "cpu"},
                    "model": {
                        "type": "tiny_gpt",
                        "vocab_size": 32,
                        "hidden_size": 8,
                        "num_layers": 1,
                        "num_heads": 2,
                        "max_position_embeddings": 8,
                    },
                    "compression": {
                        "quantization": [
                            {
                                "id": "body_int8",
                                "type": "int8",
                                "selector": {"include": ["blocks.*"]},
                            },
                            {
                                "id": "head_int4",
                                "type": "int4",
                                "group_size": 4,
                                "selector": {"include": ["lm_head"]},
                            },
                        ]
                    },
                    "pipeline": {
                        "stages": [
                            "runtime_setup",
                            "build_model",
                            "quantize_model",
                        ]
                    },
                }
            )

        report = result["metrics"]["quantization"]
        methods = {module["metadata"]["quantizer"] for module in report["modules"]}
        self.assertEqual(result["metrics"]["quantization_pass_count"], 2)
        self.assertEqual(methods, {"SymmetricInt8Quantizer", "GroupwiseInt4Quantizer"})

    def test_model_io_preserves_multimodal_keyword_inputs(self) -> None:
        batch = {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "pixel_values": torch.randn(2, 3, 8, 8),
            "labels": torch.ones(2, 4, dtype=torch.long),
            "_sample_id": ["a", "b"],
        }
        inputs, labels = prepare_model_inputs(batch, torch.device("cpu"))

        self.assertEqual(set(inputs), {"input_ids", "pixel_values"})
        self.assertEqual(labels.shape, (2, 4))
        logits = torch.randn(2, 4, 32)
        self.assertIs(extract_logits({"logits": logits}), logits)


if __name__ == "__main__":
    unittest.main()
