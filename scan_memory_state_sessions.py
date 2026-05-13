"""Scan adaptive_memory and dialog_state directories for potential session conflicts."""

from __future__ import annotations

import argparse
import json
import sys

from dialog_state_manager import (
    STATE_DIR,
    build_memory_conflict_event,
    detect_memory_state_conflicts,
    format_memory_conflict_event,
    load_dialog_state,
)
from memory_manager import MEMORY_DIR, load_session


def _build_args():
    parser = argparse.ArgumentParser(
        description="Scan adaptive_memory and dialog_state for cross-layer conflicts."
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Show sessions without detected conflicts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a single JSON summary instead of line-oriented text.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Skip thread ids starting with the given prefix. Can be used multiple times.",
    )
    return parser.parse_args()


def scan_sessions(exclude_prefixes: list[str] | None = None) -> list[dict]:
    exclude_prefixes = exclude_prefixes or []
    thread_ids = set()
    if MEMORY_DIR.exists():
        thread_ids.update(path.stem for path in MEMORY_DIR.glob("*.json"))
    if STATE_DIR.exists():
        thread_ids.update(path.stem for path in STATE_DIR.glob("*.json"))

    results = []
    for thread_id in sorted(thread_ids):
        if any(thread_id.startswith(prefix) for prefix in exclude_prefixes):
            continue
        memory_session = load_session(thread_id)
        dialog_state = load_dialog_state(thread_id)
        conflicts = detect_memory_state_conflicts(memory_session, dialog_state)
        results.append(
            build_memory_conflict_event(
                thread_id=thread_id,
                conflicts=conflicts,
                dialog_state=dialog_state,
                memory_session=memory_session,
                effective_query=dialog_state.get("last_effective_query"),
            )
        )
    return results


def _print_text(results: list[dict], include_clean: bool) -> int:
    shown = 0
    conflict_sessions = 0
    for event in results:
        if event.get("conflict_count", 0) == 0:
            if include_clean:
                print(
                    "[memory_scan] "
                    f"thread_id={event['thread_id']} conflict_count=0 status=clean"
                )
                shown += 1
            continue
        print(format_memory_conflict_event(event))
        shown += 1
        conflict_sessions += 1
    print(
        "[memory_scan_summary] "
        f"scanned={len(results)} shown={shown} conflict_sessions={conflict_sessions}"
    )
    return 0


def main() -> int:
    args = _build_args()
    results = scan_sessions(exclude_prefixes=args.exclude_prefix)
    if args.json:
        summary = {
            "scanned": len(results),
            "conflict_sessions": sum(1 for item in results if item.get("conflict_count", 0) > 0),
            "results": results if args.include_clean else [item for item in results if item.get("conflict_count", 0) > 0],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    return _print_text(results, include_clean=args.include_clean)


if __name__ == "__main__":
    sys.exit(main())
