"""Process-isolated sandbox scheduler for MCP tasks."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

_MAX_WORKERS = 2
_SEMAPHORE = threading.BoundedSemaphore(_MAX_WORKERS)

_FAIL_COUNT = 0
_OPEN_UNTIL = 0.0
_FAIL_THRESHOLD = 5
_OPEN_SECONDS = 30


def run_mcp_task(payload: dict, timeout_seconds: int = 90) -> dict:
    """Run one MCP task in a child Python process with timeout and circuit-breaker."""
    global _FAIL_COUNT, _OPEN_UNTIL

    now = time.time()
    if now < _OPEN_UNTIL:
        return {
            "ok": False,
            "source": "sandbox",
            "error": f"circuit_open_until={_OPEN_UNTIL:.0f}",
        }

    acquired = _SEMAPHORE.acquire(timeout=1.0)
    if not acquired:
        return {
            "ok": False,
            "source": "sandbox",
            "error": "sandbox_busy",
        }

    try:
        worker_path = Path(__file__).with_name("mcp_worker.py")
        command = [
            sys.executable,
            str(worker_path),
            "--payload",
            json.dumps(payload, ensure_ascii=False),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _record_failure()
        return {
            "ok": False,
            "source": "sandbox",
            "error": f"timeout_after_{timeout_seconds}s",
        }
    finally:
        _SEMAPHORE.release()

    parsed = _extract_json_from_output(result.stdout)
    if result.returncode != 0:
        _record_failure()
        return {
            "ok": False,
            "source": "sandbox",
            "error": f"worker_exit_{result.returncode}: {result.stderr.strip()[:400]}",
            "raw_stdout": result.stdout[-500:],
        }

    if not parsed:
        _record_failure()
        return {
            "ok": False,
            "source": "sandbox",
            "error": "worker_output_not_json",
            "raw_stdout": result.stdout[-500:],
        }

    if not parsed.get("ok"):
        _record_failure()
        return parsed

    _FAIL_COUNT = 0
    return parsed


def _record_failure():
    global _FAIL_COUNT, _OPEN_UNTIL
    _FAIL_COUNT += 1
    if _FAIL_COUNT >= _FAIL_THRESHOLD:
        _OPEN_UNTIL = time.time() + _OPEN_SECONDS
        _FAIL_COUNT = 0


def _extract_json_from_output(stdout: str) -> dict | None:
    if not stdout:
        return None
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            continue
    return None
