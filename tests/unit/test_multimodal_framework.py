"""Behavioral tests for multimodal components, models, data, and training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from compression import ModuleSelector, quantize_linear_modules
from compression.quantization.int8 import SymmetricInt8Quantizer
from compression.quantization.quant_linear import ReferenceQuantizedLinear
from core.config import load_config
from data import build_dataset
from experiments import ExperimentConfigError, normalize_experiment_config, run_experiment
from models import build_model
from modules.multimodal import (
    MULTIMODAL_FUSION_REGISTRY,
    MULTIMODAL_PROJECTOR_REGISTRY,
    MULTIMODAL_RESAMPLER_REGISTRY,
    ModalityFeatures,
    build_multimodal_fusion,
    build_multimodal_resampler,
)
from modules.vision import VISION_ENCODER_REGISTRY, build_vision_encoder
from training import build_loss


def _tiny_vlm(**overrides):
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 4,
        "max_position_embeddings": 6,
        "image_size": 8,
        "patch_size": 4,
        "vision_hidden_size": 12,
        "vision_num_layers": 1,
        "vision_num_heads": 3,
        "projector_type": {"type": "mlp", "hidden_size": 16},
        "resampler_type": {"type": "adaptive_pool", "num_tokens": 2},
    }
    config.update(overrides)
    return build_model("tiny_vlm", **config)


class MultimodalComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)

    def test_multimodal_component_registries_are_discoverable(self) -> None:
        self.assertIn("patch_transformer", VISION_ENCODER_REGISTRY)
        self.assertEqual(
            MULTIMODAL_PROJECTOR_REGISTRY.canonical_names(),
            ("linear", "mlp"),
        )
        self.assertEqual(
            MULTIMODAL_RESAMPLER_REGISTRY.canonical_names(),
            ("adaptive_pool", "identity"),
        )
        self.assertIn("prefix", MULTIMODAL_FUSION_REGISTRY)

    def test_patch_encoder_handles_multiple_images_and_masks(self) -> None:
        encoder = build_vision_encoder(
            "patch_transformer",
            image_size=8,
            patch_size=4,
            hidden_size=12,
            num_layers=1,
            num_heads=3,
            max_images=2,
        )
        images = torch.randn(2, 2, 3, 8, 8)
        image_mask = torch.tensor([[True, False], [True, True]])

        features = encoder(images, image_mask=image_mask)

        self.assertEqual(features.embeddings.shape, (2, 8, 12))
        self.assertEqual(features.valid_mask().sum(dim=1).tolist(), [4, 8])
        self.assertTrue(torch.isfinite(features.embeddings).all())
        self.assertTrue(torch.equal(features.embeddings[0, 4:], torch.zeros(4, 12)))

    def test_masked_adaptive_resampler_preserves_empty_bins(self) -> None:
        features = ModalityFeatures(
            torch.tensor([[[1.0], [3.0], [9.0], [11.0]]]),
            torch.tensor([[True, True, False, False]]),
        )
        resampler = build_multimodal_resampler(
            {"type": "adaptive_pool", "num_tokens": 2}
        )

        output = resampler(features)

        self.assertEqual(output.embeddings.shape, (1, 2, 1))
        self.assertEqual(output.valid_mask().tolist(), [[True, False]])
        self.assertAlmostEqual(float(output.embeddings[0, 0, 0]), 2.0)
        self.assertEqual(float(output.embeddings[0, 1, 0]), 0.0)

    def test_prefix_fusion_returns_text_positions(self) -> None:
        text = torch.randn(2, 3, 8)
        modality = ModalityFeatures(torch.randn(2, 2, 8))
        fusion = build_multimodal_fusion("prefix")

        output = fusion(text, modality)

        self.assertEqual(output.embeddings.shape, (2, 5, 8))
        self.assertEqual(output.text_positions.tolist(), [[2, 3, 4], [2, 3, 4]])
        self.assertTrue(output.attention_mask.all())


class TinyVLMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)

    def test_tiny_vlm_forward_and_backward_reach_vision_and_projector(self) -> None:
        model = _tiny_vlm()
        input_ids = torch.randint(0, 32, (2, 6))
        labels = torch.randint(0, 32, (2, 6))
        pixel_values = torch.randn(2, 1, 3, 8, 8)

        output = model(input_ids=input_ids, pixel_values=pixel_values)
        loss = build_loss("causal_lm")(output.logits, labels=labels)
        loss.backward()

        self.assertEqual(output.logits.shape, (2, 6, 32))
        self.assertEqual(output.modality_token_count.tolist(), [2, 2])
        self.assertIsNotNone(model.vision_encoder.patch_embedding.weight.grad)
        projector_parameter = next(model.multimodal_projector.parameters())
        self.assertIsNotNone(projector_parameter.grad)

    def test_freeze_controls_leave_only_projector_trainable(self) -> None:
        model = _tiny_vlm(
            freeze_vision_encoder=True,
            freeze_language_model=True,
        )
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("multimodal_projector.") for name in trainable)
        )

    def test_tiny_vlm_state_dict_round_trip(self) -> None:
        source = _tiny_vlm().eval()
        target = _tiny_vlm().eval()
        target.load_state_dict(source.state_dict())
        inputs = {
            "input_ids": torch.randint(0, 32, (1, 6)),
            "pixel_values": torch.randn(1, 1, 3, 8, 8),
        }

        with torch.no_grad():
            source_logits = source(**inputs).logits
            target_logits = target(**inputs).logits

        self.assertTrue(torch.equal(source_logits, target_logits))

    def test_projector_scope_quantization_does_not_touch_other_towers(self) -> None:
        model = _tiny_vlm()
        transformed, report = quantize_linear_modules(
            model,
            SymmetricInt8Quantizer(),
            selector=ModuleSelector(scopes=("projector",)),
        )

        self.assertTrue(report.records)
        self.assertTrue(
            all(record.name.startswith("multimodal_projector.") for record in report.records)
        )
        self.assertFalse(
            any(
                isinstance(module, ReferenceQuantizedLinear)
                for module in transformed.vision_encoder.modules()
            )
        )
        output = transformed(
            input_ids=torch.randint(0, 32, (1, 6)),
            pixel_values=torch.randn(1, 1, 3, 8, 8),
        )
        self.assertEqual(output.logits.shape, (1, 6, 32))


class MultimodalDataAndPipelineTest(unittest.TestCase):
    def test_config_rejects_mismatched_multimodal_shapes(self) -> None:
        with self.assertRaisesRegex(ExperimentConfigError, "image_size must match"):
            normalize_experiment_config(
                {
                    "model": {
                        "type": "tiny_vlm",
                        "image_size": 8,
                        "patch_size": 4,
                        "max_images": 1,
                    },
                    "data": {
                        "dataset": {
                            "type": "synthetic_vision_language",
                            "image_size": 16,
                            "num_images": 1,
                        }
                    },
                }
            )

    def test_synthetic_dataset_is_deterministic_and_image_conditioned(self) -> None:
        dataset = build_dataset(
            {
                "type": "synthetic_vision_language",
                "vocab_size": 32,
                "num_samples": 4,
                "sequence_length": 6,
                "image_size": 8,
                "num_classes": 4,
                "seed": 17,
            }
        )

        first = dataset[3]
        repeated = dataset[3]
        self.assertTrue(torch.equal(first["pixel_values"], repeated["pixel_values"]))
        self.assertTrue(torch.equal(first["input_ids"], repeated["input_ids"]))
        self.assertEqual(int(first["labels"][-1]), 3)
        self.assertEqual(first["pixel_values"].shape, (1, 3, 8, 8))

    def test_tiny_vlm_config_runs_end_to_end_and_tracks_components(self) -> None:
        config = load_config("configs/multimodal/tiny_vlm_smoke.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            config["experiment"]["output_dir"] = temporary
            config["training"]["max_steps"] = 1
            result = run_experiment(config)
            manifest = json.loads(
                (Path(result["run_dir"]) / "manifest.json").read_text()
            )
            checkpoint_exists = Path(result["metrics"]["checkpoint_path"]).exists()

        self.assertEqual(result["metrics"]["steps"], 1)
        self.assertEqual(result["metrics"]["modality_tokens_seen"], 4)
        self.assertTrue(
            torch.isfinite(torch.tensor(result["metrics"]["final_train_loss"]))
        )
        self.assertTrue(checkpoint_exists)
        self.assertIn("vision_encoder", manifest["components"])
        self.assertIn("multimodal_projector", manifest["components"])
        self.assertIn("multimodal_resampler", manifest["components"])
        self.assertIn("multimodal_fusion", manifest["components"])


if __name__ == "__main__":
    unittest.main()
