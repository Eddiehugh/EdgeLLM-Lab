"""Private local connection profiles for replaceable remote machines."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
PROFILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
CONNECTION_KEYS = {
    "host",
    "user",
    "port",
    "identity_file",
    "password",
    "ssh_options",
}
REDACTED_VALUE = "********"


def parse_ssh_command(command: str) -> dict[str, Any]:
    """Parse a login-only OpenSSH command into reusable connection fields."""

    tokens = shlex.split(command)
    if tokens and Path(tokens[0]).name == "ssh":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("SSH command is empty")

    profile: dict[str, Any] = {}
    ssh_options: list[str] = []
    target: str | None = None
    index = 0
    value_options = {"-o", "-F", "-J"}
    passthrough_flags = {"-4", "-6", "-A", "-C", "-q", "-v"}
    while index < len(tokens):
        token = tokens[index]
        if target is not None:
            raise ValueError("SSH profile command must not contain a remote command")
        if token == "--":
            index += 1
            if index >= len(tokens):
                raise ValueError("SSH command is missing a host")
            target = tokens[index]
        elif token in {"-p", "-i", "-l"}:
            index += 1
            if index >= len(tokens):
                raise ValueError(f"SSH option {token} requires a value")
            value = tokens[index]
            key = {"-p": "port", "-i": "identity_file", "-l": "user"}[token]
            profile[key] = int(value) if key == "port" else value
        elif token in value_options:
            index += 1
            if index >= len(tokens):
                raise ValueError(f"SSH option {token} requires a value")
            ssh_options.extend([token, tokens[index]])
        elif token in passthrough_flags:
            ssh_options.append(token)
        elif token.startswith("-"):
            raise ValueError(
                f"Unsupported SSH option '{token}'. Use direct profile fields or "
                "an -o option for advanced OpenSSH settings."
            )
        else:
            target = token
        index += 1

    if target is None:
        raise ValueError("SSH command is missing a host")
    if "@" in target:
        target_user, host = target.rsplit("@", 1)
        if profile.get("user") and profile["user"] != target_user:
            raise ValueError("Conflicting SSH users in -l and user@host")
        profile["user"] = target_user
        profile["host"] = host
    else:
        profile["host"] = target
    if ssh_options:
        profile["ssh_options"] = ssh_options
    return _validate_connection(profile)


def _validate_connection(profile: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(profile) - CONNECTION_KEYS)
    if unknown:
        raise ValueError(f"Unsupported connection profile fields: {', '.join(unknown)}")
    result = dict(profile)
    host = result.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Connection profile host must be a non-empty string")
    result["host"] = host.strip()
    if "user" in result:
        result["user"] = str(result["user"]).strip()
    if "port" in result:
        port = int(result["port"])
        if not 1 <= port <= 65535:
            raise ValueError("Connection profile port must be between 1 and 65535")
        result["port"] = port
    if "identity_file" in result:
        result["identity_file"] = str(result["identity_file"])
    if "password" in result:
        password = result["password"]
        if not isinstance(password, str) or not password:
            raise ValueError("Connection profile password must be a non-empty string")
    if "ssh_options" in result:
        options = result["ssh_options"]
        if not isinstance(options, list) or not all(
            isinstance(value, str) for value in options
        ):
            raise TypeError("Connection profile ssh_options must be a list of strings")
    return result


def redact_connection(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a display-safe copy of a connection profile."""

    result = dict(profile)
    if "password" in result:
        result["password"] = REDACTED_VALUE
    return result


class ConnectionProfileStore:
    """Persist SSH endpoints outside version-controlled experiment configs."""

    def __init__(self, path: str | Path = ".edgellm/connections.json"):
        self.path = Path(path).expanduser().resolve()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported connection profile schema in {self.path}: "
                f"{payload.get('schema_version')!r}"
            )
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError(f"Invalid connection profile store: {self.path}")
        result: dict[str, dict[str, Any]] = {}
        for raw_name, value in profiles.items():
            name = self._validate_name(str(raw_name))
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Connection profile '{name}' must be an object in {self.path}"
                )
            result[name] = _validate_connection(value)
        return result

    def _write(self, profiles: Mapping[str, Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "profiles": dict(profiles),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not PROFILE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Connection profile name must contain only letters, numbers, '.', "
                "'_' or '-', and must start with a letter or number"
            )
        return name

    def get(self, name: str) -> dict[str, Any]:
        name = self._validate_name(name)
        try:
            return dict(self._load()[name])
        except KeyError as exc:
            raise KeyError(
                f"Unknown connection profile '{name}' in {self.path}. "
                "Create it with `edgellm connection set`."
            ) from exc

    def list(self) -> dict[str, dict[str, Any]]:
        return {name: dict(value) for name, value in sorted(self._load().items())}

    def set(self, name: str, values: Mapping[str, Any]) -> dict[str, Any]:
        name = self._validate_name(name)
        profiles = self._load()
        merged = dict(profiles.get(name, {}))
        for key, value in values.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        profile = _validate_connection(merged)
        profiles[name] = profile
        self._write(profiles)
        return dict(profile)

    def remove(self, name: str) -> None:
        name = self._validate_name(name)
        profiles = self._load()
        if name not in profiles:
            raise KeyError(f"Unknown connection profile '{name}' in {self.path}")
        del profiles[name]
        self._write(profiles)

    def resolve(self, executor_config: Mapping[str, Any]) -> dict[str, Any]:
        """Merge a named profile under stable executor settings."""

        config = dict(executor_config)
        profile_name = config.pop("profile", None)
        if profile_name is None:
            return config
        profile = self.get(str(profile_name))
        return {**profile, **config}
