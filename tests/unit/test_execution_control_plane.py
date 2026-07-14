from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from execution import CommandSpec, JobSpec, JobState, RunManager, WorkloadSpec
from execution.executors import build_executor
from execution.executors.clearml import ClearMLExecutor
from execution.executors.huggingface_jobs import HuggingFaceJobsExecutor
from execution.metadata import JsonMetadataStore
from execution.runtimes import build_runtime
from execution.specs import ArtifactSpec, RuntimeSpec, SourceSpec
from execution.worker import run_worker


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def tiny_config(root: Path) -> dict:
    return {
        "experiment": {"name": "execution-smoke"},
        "runtime": {"device": "cpu", "seed": 7},
        "model": {
            "name": "tiny_gpt",
            "vocab_size": 256,
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "max_position_embeddings": 8,
        },
        "data": {
            "tokenizer_type": "char",
            "text": "EdgeLLM execution test.\n" * 8,
            "block_size": 8,
        },
        "loss": {"type": "causal_lm"},
        "training": {
            "batch_size": 1,
            "max_steps": 1,
            "optimizer": {"type": "adamw"},
            "scheduler": {"type": "constant"},
        },
        "execution": {
            "executor": {"type": "local"},
            "runtime": {"type": "native"},
            "artifact_store": {
                "type": "local",
                "root": str(root / "artifacts"),
            },
        },
    }


class ExecutionControlPlaneTest(unittest.TestCase):
    def test_job_spec_round_trip(self) -> None:
        spec = JobSpec(
            job_id="job-1",
            name="round-trip",
            experiment_config={"experiment": {"name": "round-trip"}},
            executor_type="local",
            executor_config={},
            runtime=RuntimeSpec(),
            artifact_store=ArtifactSpec(),
            source=SourceSpec(project_root=str(PROJECT_ROOT)),
            workspace="/tmp/job-1",
        )
        self.assertEqual(JobSpec.from_dict(spec.to_dict()), spec)

    def test_external_workload_clones_runs_and_collects_declared_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external_repo = root / "external"
            external_repo.mkdir()
            subprocess.run(
                ["git", "init"],
                cwd=external_repo,
                check=True,
                capture_output=True,
            )
            (external_repo / "run.py").write_text(
                "from pathlib import Path\n"
                "Path('checkpoint').mkdir()\n"
                "Path('checkpoint/model.txt').write_text('trained\\n')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "run.py"], cwd=external_repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=EdgeLLM Test",
                    "-c",
                    "user.email=test@edgellm.local",
                    "commit",
                    "-m",
                    "test workload",
                ],
                cwd=external_repo,
                check=True,
                capture_output=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=external_repo,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            spec = JobSpec(
                job_id="external-job",
                name="external-test",
                experiment_config={"experiment": {"name": "external-test"}},
                executor_type="local",
                executor_config={},
                runtime=RuntimeSpec(),
                artifact_store=ArtifactSpec(
                    type="local", config={"root": str(root / "published")}
                ),
                source=SourceSpec(project_root=str(PROJECT_ROOT)),
                workspace=str(root / "workspace"),
                workload=WorkloadSpec(
                    type="external_project",
                    integration="fixture",
                    source=SourceSpec(
                        repo_url=str(external_repo),
                        revision=revision,
                        project_root=str(external_repo),
                    ),
                    command=CommandSpec(argv=(sys.executable, "run.py")),
                    artifacts=("checkpoint",),
                ),
            )

            result = run_worker(spec)

            self.assertEqual(result["state"], JobState.COMPLETED.value)
            published = Path(result["artifact_uri"])
            self.assertEqual(
                (published / "artifacts" / "checkpoint" / "model.txt").read_text(),
                "trained\n",
            )
            metadata = json.loads((published / "external-run.json").read_text())
            self.assertEqual(metadata["revision"], revision)
            self.assertIn("Running in", (published / "command.log").read_text())

    def test_local_job_runs_and_fetches_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RunManager(
                metadata_store=JsonMetadataStore(root / "metadata"),
                project_root=PROJECT_ROOT,
            )
            record = manager.submit(tiny_config(root))
            completed = manager.wait(record.job_id, poll_interval=0.05, timeout=30)
            self.assertEqual(completed.state, JobState.COMPLETED)
            artifact = Path(completed.artifact_uri or "")
            self.assertTrue((artifact / "metrics.json").exists())
            fetched = manager.fetch(record.job_id, root / "download")
            self.assertTrue((fetched / "manifest.json").exists())
            self.assertIn("status", json.loads((fetched / "metrics.json").read_text()))

    def test_colab_executor_generates_notebook_without_cloud_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RunManager(
                metadata_store=JsonMetadataStore(root / "metadata"),
                project_root=PROJECT_ROOT,
            )
            config = tiny_config(root)
            config["execution"]["executor"] = {"type": "colab"}
            config["execution"]["source"] = {"require_clean": False}
            record = manager.submit(config)
            self.assertEqual(record.state, JobState.PREPARED)
            notebook = Path(record.details["notebook_path"])
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            self.assertIn("execution.worker", json.dumps(payload))

    def test_ephemeral_remote_provider_rejects_local_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = RunManager(
                metadata_store=JsonMetadataStore(Path(temporary) / "metadata"),
                project_root=PROJECT_ROOT,
            )
            config = tiny_config(Path(temporary))
            config["execution"]["executor"] = {"type": "huggingface_jobs"}
            config["execution"]["source"] = {"require_clean": False}
            with self.assertRaisesRegex(ValueError, "durable remote artifact store"):
                manager.build_spec(config)

    def test_executor_and_runtime_factories_are_lazy(self) -> None:
        self.assertEqual(type(build_executor("autodl")).__name__, "SSHExecutor")
        self.assertEqual(type(build_executor("clearml")).__name__, "ClearMLExecutor")
        runtime = build_runtime(
            RuntimeSpec(type="docker", image="python:3.11", python="python3")
        )
        command = runtime.worker_command("/repo", "/job", "/job/spec.json")
        self.assertEqual(command[:3], ["docker", "run", "--rm"])

    def test_huggingface_jobs_adapter_uses_official_lifecycle_api(self) -> None:
        calls = {}

        class FakeHub:
            @staticmethod
            def run_job(**kwargs):
                calls["submit"] = kwargs
                return SimpleNamespace(id="hf-1", url="https://hf.example/jobs/hf-1")

            @staticmethod
            def inspect_job(job_id, **kwargs):
                calls["inspect"] = (job_id, kwargs)
                return SimpleNamespace(
                    status=SimpleNamespace(stage="COMPLETED", message=None)
                )

            @staticmethod
            def fetch_job_logs(job_id, **kwargs):
                calls["logs"] = (job_id, kwargs)
                return ["first", "second"]

            @staticmethod
            def cancel_job(job_id, **kwargs):
                calls["cancel"] = (job_id, kwargs)

        spec = JobSpec(
            job_id="hf-job",
            name="hf-test",
            experiment_config={"experiment": {"name": "hf-test"}},
            executor_type="huggingface_jobs",
            executor_config={},
            runtime=RuntimeSpec(type="docker", image="python:3.12"),
            artifact_store=ArtifactSpec(
                type="huggingface_hub", config={"repo_id": "user/artifacts"}
            ),
            source=SourceSpec(
                repo_url="https://github.com/example/repo.git",
                revision="abc123",
                project_root=str(PROJECT_ROOT),
            ),
            workspace="/tmp/hf-job",
        )
        executor = HuggingFaceJobsExecutor({"flavor": "cpu-basic"})
        with mock.patch.object(executor, "_hub", return_value=FakeHub):
            record = executor.submit(spec)
            self.assertEqual(record.provider_job_id, "hf-1")
            self.assertIn("execution.worker", calls["submit"]["command"][2])
            self.assertEqual(executor.logs(record), "first\nsecond")
            completed = executor.status(record)
            self.assertEqual(completed.state, JobState.COMPLETED)

    def test_clearml_adapter_creates_and_enqueues_pinned_task(self) -> None:
        calls = {}

        class FakeTask:
            id = "clearml-1"
            status = "in_progress"

            @classmethod
            def create(cls, **kwargs):
                calls["create"] = kwargs
                return cls()

            @classmethod
            def enqueue(cls, **kwargs):
                calls["enqueue"] = kwargs

            @classmethod
            def get_task(cls, task_id):
                calls["get_task"] = task_id
                return cls()

            @staticmethod
            def get_reported_console_output(number_of_reports):
                return [f"reports={number_of_reports}", "training"]

            @staticmethod
            def get_output_log_web_page():
                return "https://clearml.example/task/clearml-1"

        spec = JobSpec(
            job_id="clearml-job",
            name="clearml-test",
            experiment_config={"experiment": {"name": "clearml-test"}},
            executor_type="clearml",
            executor_config={},
            runtime=RuntimeSpec(type="docker", image="python:3.12"),
            artifact_store=ArtifactSpec(
                type="huggingface_hub", config={"repo_id": "user/artifacts"}
            ),
            source=SourceSpec(
                repo_url="https://github.com/example/repo.git",
                revision="abc123",
                project_root=str(PROJECT_ROOT),
            ),
            workspace="/tmp/clearml-job",
        )
        executor = ClearMLExecutor({"queue": "gpu", "packages": ["PyYAML"]})
        with mock.patch.object(executor, "_task_class", return_value=FakeTask):
            record = executor.submit(spec)
            self.assertEqual(calls["create"]["commit"], "abc123")
            self.assertEqual(calls["create"]["script"], "execution/worker.py")
            self.assertEqual(calls["enqueue"]["queue_name"], "gpu")
            self.assertEqual(executor.status(record).state, JobState.RUNNING)
            self.assertIn("training", executor.logs(record))


if __name__ == "__main__":
    unittest.main()
