"""Runtime dependency checks for predictable startup behavior."""

from __future__ import annotations

import importlib.util
import os
import shutil


def _missing_packages(packages: list[str]) -> list[str]:
    return [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]


def _missing_env_vars(env_vars: list[str]) -> list[str]:
    missing = []
    for env_var in env_vars:
        value = os.getenv(env_var, "").strip()
        if not value:
            missing.append(env_var)
    return missing


def _missing_commands(commands: list[str]) -> list[str]:
    return [cmd for cmd in commands if shutil.which(cmd) is None]


def assert_runtime_ready(
    *,
    stage: str,
    packages: list[str] | None = None,
    env_vars: list[str] | None = None,
    commands: list[str] | None = None,
):
    """Raise RuntimeError with actionable messages when runtime is not ready."""
    packages = packages or []
    env_vars = env_vars or []
    commands = commands or []

    missing_pkgs = _missing_packages(packages)
    missing_env = _missing_env_vars(env_vars)
    missing_cmds = _missing_commands(commands)

    issues = []
    if missing_pkgs:
        issues.append(f"missing packages: {', '.join(missing_pkgs)}")
    if missing_env:
        issues.append(f"missing env vars: {', '.join(missing_env)}")
    if missing_cmds:
        issues.append(f"missing commands: {', '.join(missing_cmds)}")

    if issues:
        joined = " | ".join(issues)
        raise RuntimeError(f"[{stage}] runtime check failed: {joined}")

