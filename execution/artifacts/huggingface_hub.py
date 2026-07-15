"""Hugging Face Hub artifact store."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from execution.artifacts.base import ArtifactStore


class HuggingFaceHubArtifactStore(ArtifactStore):
    def __init__(
        self,
        repo_id: str,
        repo_type: str = "dataset",
        path_prefix: str = "edgellm-jobs",
        token_env: str = "HF_TOKEN",
        private: bool = True,
    ):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.path_prefix = path_prefix.strip("/")
        self.token_env = token_env
        self.private = private

    @staticmethod
    def _hub():
        try:
            import huggingface_hub
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face artifacts require the 'hf' extra: "
                "pip install -e '.[hf]'"
            ) from exc
        return huggingface_hub

    def _path(self, job_id: str) -> str:
        return f"{self.path_prefix}/{job_id}" if self.path_prefix else job_id

    def uri_for(self, job_id: str) -> str:
        return f"hf://{self.repo_type}/{self.repo_id}/{self._path(job_id)}"

    def publish(self, source: str | Path, job_id: str) -> str:
        hub = self._hub()
        api = hub.HfApi(token=os.environ.get(self.token_env))
        api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            private=self.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(Path(source).resolve()),
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            path_in_repo=self._path(job_id),
        )
        return self.uri_for(job_id)

    def fetch(self, uri: str, destination: str | Path) -> Path:
        prefix = f"hf://{self.repo_type}/{self.repo_id}/"
        if not uri.startswith(prefix):
            raise ValueError(f"Artifact URI does not belong to {self.repo_id}: {uri}")
        path_in_repo = uri[len(prefix) :]
        hub = self._hub()
        snapshot = Path(
            hub.snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=os.environ.get(self.token_env),
            )
        )
        source = snapshot / path_in_repo
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"Artifact destination already exists: {target}")
        shutil.copytree(source, target)
        return target
