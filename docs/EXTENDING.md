# Extending EdgeLLM-Lab

The framework has two extension surfaces. Add a component when introducing a
replaceable implementation. Add an experiment stage when introducing workflow
behavior such as evaluation, quantization, export, or device benchmarking.

## Add a Component

Register the implementation in its domain registry and declare honest metadata:

```python
from core import Maturity, ProjectLevel
from modules.attention import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "my_attention",
    level=ProjectLevel.LEARN,
    maturity=Maturity.EXPERIMENTAL,
    capabilities=("self_attention", "training"),
)
class MyAttention:
    ...
```

Use `planned` for a nonfunctional adapter, `experimental` for an executable
implementation, `verified` after correctness tests, and `production` only after
target-device and regression validation.

Keep a simple technique in one file. Promote it to a package when it develops
multiple kernels, configs, state formats, or converters. Preserve imports in
the package `__init__.py` during that promotion.

## Add a Multimodal Component

Keep modality encoding, token resampling, projection, and fusion independent.
For example, a new vision encoder returns the shared feature contract and
declares its maximum token budget:

```python
import torch.nn as nn

from modules.multimodal import ModalityFeatures
from modules.vision.encoder import VISION_ENCODER_REGISTRY


@VISION_ENCODER_REGISTRY.register("my_vision_encoder")
class MyVisionEncoder(nn.Module):
    max_output_tokens = 64

    def forward(self, pixel_values, image_mask=None):
        embeddings, token_mask = encode(pixel_values, image_mask)
        return ModalityFeatures(embeddings, token_mask)
```

Register projectors under `modules/multimodal/projector/`, resamplers under
`resampler/`, and fusion methods under `fusion/`. A model family in
`models/multimodal/` composes them and exposes `component_selections()` plus
`compression_scopes()`. Do not put image preprocessing in the model or fusion
logic in the experiment runner.

## Add a Pipeline Stage

```python
from core import Maturity, ProjectLevel
from experiments import ExperimentStage, STAGE_REGISTRY


@STAGE_REGISTRY.register(
    "evaluate",
    level=ProjectLevel.EXPERIMENT,
    maturity=Maturity.EXPERIMENTAL,
)
class EvaluateStage(ExperimentStage):
    requires = ("model", "dataloader", "device")

    def run(self, context):
        model = context.require("model")
        context.metrics["validation_loss"] = evaluate(model)
```

Load the module and place the stage in the experiment config:

```yaml
imports:
  - my_lab.evaluation

pipeline:
  stages:
    - runtime_setup
    - build_data
    - build_model
    - build_training
    - train
    - evaluate
    - model_stats
    - checkpoint
```

Standalone research files, including paper-specific implementations, can be
loaded without turning their directory into an installable package:

```yaml
extensions:
  paths:
    - paper_reproductions/my-paper/implementation/components.py
```

The stage should place reusable runtime objects in `context.objects`, scalar or
structured measurements in `context.metrics`, and generated files in the run
directory through `context.run`. Register every generated file with
`context.record_artifact`.

## Validate Before Running

```bash
python3 -m cli validate-config -c configs/my_experiment.yaml
python3 -m cli list-components --details
python3 -m cli component-info experiment_stage evaluate
```

Do not add model-specific branches to `ExperimentRunner`, import optional
external projects from the core framework, or report a placeholder backend as
implemented.
