"""Artifact store implementations and factory."""

from execution.artifacts.base import ArtifactStore
from execution.artifacts.huggingface_hub import HuggingFaceHubArtifactStore
from execution.artifacts.local import LocalArtifactStore
from execution.artifacts.s3 import S3ArtifactStore
from execution.specs import ArtifactSpec


def build_artifact_store(spec: ArtifactSpec) -> ArtifactStore:
    factories = {
        "local": LocalArtifactStore,
        "s3": S3ArtifactStore,
        "huggingface_hub": HuggingFaceHubArtifactStore,
        "hf_hub": HuggingFaceHubArtifactStore,
    }
    try:
        factory = factories[spec.type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown artifact store '{spec.type}'. Available: {', '.join(factories)}"
        ) from exc
    return factory(**spec.config)


__all__ = [
    "ArtifactStore",
    "HuggingFaceHubArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "build_artifact_store",
]
