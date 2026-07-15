"""Tests for replaceable AutoDL SSH connection profiles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from execution import (
    ConnectionProfileStore,
    RunManager,
    parse_ssh_command,
    redact_connection,
)
from execution.executors.ssh import SSHExecutor
from execution.metadata import JsonMetadataStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConnectionProfileTest(unittest.TestCase):
    def test_password_is_stored_privately_and_redacted_for_display(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "connections.json"
            store = ConnectionProfileStore(path)
            stored = store.set(
                "autodl-main",
                {
                    "host": "example.com",
                    "user": "root",
                    "password": "local-secret",
                },
            )

            self.assertEqual(store.get("autodl-main")["password"], "local-secret")
            self.assertEqual(redact_connection(stored)["password"], "********")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_password_auth_uses_sshpass_without_putting_secret_in_argv(self) -> None:
        executor = SSHExecutor(
            {
                "host": "example.com",
                "user": "root",
                "password": "local-secret",
            }
        )
        completed = mock.Mock(stdout="ok")
        with mock.patch(
            "execution.executors.ssh.shutil.which", return_value="/sshpass"
        ):
            with mock.patch(
                "execution.executors.ssh.subprocess.run", return_value=completed
            ) as run:
                self.assertEqual(executor._ssh("true"), "ok")

        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["/sshpass", "-e", "ssh"])
        self.assertNotIn("local-secret", argv)
        self.assertEqual(run.call_args.kwargs["env"]["SSHPASS"], "local-secret")

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

    def test_password_is_referenced_but_not_serialized_in_job_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            connections = ConnectionProfileStore(root / "connections.json")
            connections.set(
                "autodl-main",
                {
                    "host": "region.autodl.com",
                    "user": "root",
                    "port": 12345,
                    "password": "local-secret",
                },
            )
            manager = RunManager(
                metadata_store=JsonMetadataStore(root / "jobs"),
                project_root=PROJECT_ROOT,
                connection_store=connections,
            )
            spec = manager.build_spec(
                {
                    "experiment": {"name": "password-profile"},
                    "execution": {
                        "executor": {"type": "autodl", "profile": "autodl-main"},
                        "source": {"require_clean": False},
                    },
                }
            )

            self.assertNotIn("password", spec.executor_config)
            self.assertEqual(
                spec.executor_config["credential_profile"], "autodl-main"
            )
            self.assertNotIn("local-secret", str(spec.to_dict()))
            self.assertEqual(manager._executor(spec).config["password"], "local-secret")

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

    def test_remote_launch_uses_valid_background_shell_syntax(self) -> None:
        command = SSHExecutor._build_remote_launch(
            ["set -e", "cd /tmp/job/source"],
            ["python3", "-m", "execution.worker", "--job-spec", "/tmp/job.json"],
            "/tmp/job/worker.log",
        )

        self.assertIn("nohup python3 -m execution.worker", command)
        self.assertIn("& printf '\\n__EDGELLM_PID__=%s\\n' $!", command)
        self.assertNotIn("&;", command)

    def test_remote_pid_parser_ignores_bootstrap_output(self) -> None:
        output = "pip install output\nmore output\n__EDGELLM_PID__=2660"

        self.assertEqual(SSHExecutor._parse_remote_pid(output), "2660")

    def test_remote_pid_parser_rejects_unmarked_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not return"):
            SSHExecutor._parse_remote_pid("pip output\n2660")

    def test_shell_init_accepts_autodl_network_bootstrap(self) -> None:
        executor = SSHExecutor(
            {
                "host": "example.com",
                "shell_init": ["source /etc/network_turbo"],
            }
        )

        self.assertEqual(executor._shell_init(), ["source /etc/network_turbo"])

    def test_shell_init_rejects_non_string_commands(self) -> None:
        executor = SSHExecutor(
            {"host": "example.com", "shell_init": ["source profile", 42]}
        )

        with self.assertRaisesRegex(TypeError, "shell_init"):
            executor._shell_init()


if __name__ == "__main__":
    unittest.main()
