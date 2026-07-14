"""Tests for replaceable AutoDL SSH connection profiles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from execution import ConnectionProfileStore, RunManager, parse_ssh_command
from execution.executors.ssh import SSHExecutor
from execution.executors.ssh import SSHExecutor
from execution.metadata import JsonMetadataStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConnectionProfileTest(unittest.TestCase):
    def test_probe_reports_missing_identity_before_connecting(self) -> None:
        executor = SSHExecutor(
            {
                "host": "example.invalid",
                "identity_file": "/missing/edgellm-test-key",
            }
        )
        with self.assertRaisesRegex(FileNotFoundError, "identity file does not exist"):
            executor.probe()

    def test_parse_autodl_ssh_command(self):
        profile = parse_ssh_command(
            "ssh -p 35394 -i ~/.ssh/id_ed25519 root@region-1.autodl.com"
        )

        self.assertEqual(profile["host"], "region-1.autodl.com")
        self.assertEqual(profile["user"], "root")
        self.assertEqual(profile["port"], 35394)
        self.assertEqual(profile["identity_file"], "~/.ssh/id_ed25519")

    def test_store_updates_endpoint_without_losing_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "connections.json"
            store = ConnectionProfileStore(path)
            store.set(
                "autodl-main",
                {
                    "host": "old.autodl.com",
                    "port": 10001,
                    "user": "root",
                    "identity_file": "~/.ssh/id_ed25519",
                },
            )
            updated = store.set(
                "autodl-main",
                {"host": "new.autodl.com", "port": 20002},
            )

            self.assertEqual(updated["host"], "new.autodl.com")
            self.assertEqual(updated["port"], 20002)
            self.assertEqual(updated["identity_file"], "~/.ssh/id_ed25519")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_run_manager_resolves_profile_and_preserves_stable_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            connections = ConnectionProfileStore(root / "connections.json")
            connections.set(
                "autodl-main",
                {"host": "region.autodl.com", "user": "root", "port": 12345},
            )
            manager = RunManager(
                metadata_store=JsonMetadataStore(root / "jobs"),
                project_root=PROJECT_ROOT,
                connection_store=connections,
            )
            spec = manager.build_spec(
                {
                    "experiment": {"name": "profile-resolution"},
                    "training": {"max_steps": 1},
                    "execution": {
                        "executor": {
                            "type": "autodl",
                            "profile": "autodl-main",
                            "remote_root": "/root/autodl-tmp/edgellm-jobs",
                        },
                        "source": {"require_clean": False},
                    },
                }
            )

            self.assertEqual(spec.executor_config["host"], "region.autodl.com")
            self.assertEqual(spec.executor_config["port"], 12345)
            self.assertEqual(
                spec.executor_config["remote_root"],
                "/root/autodl-tmp/edgellm-jobs",
            )
            self.assertNotIn("profile", spec.executor_config)

    def test_identity_path_and_advanced_options_apply_to_ssh_and_scp(self):
        executor = SSHExecutor(
            {
                "host": "example.com",
                "identity_file": "~/.ssh/id_ed25519",
                "ssh_options": ["-o", "StrictHostKeyChecking=accept-new"],
            }
        )

        expected_identity = str(Path("~/.ssh/id_ed25519").expanduser())
        self.assertIn(expected_identity, executor._ssh_options())
        self.assertIn(expected_identity, executor._scp_options())
        self.assertIn("StrictHostKeyChecking=accept-new", executor._scp_options())


if __name__ == "__main__":
    unittest.main()
