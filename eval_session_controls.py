"""Regression checks for session reset helpers and memory priority note."""

from __future__ import annotations

import sys

from dialog_state_manager import (
    build_memory_conflict_event,
    build_memory_priority_note,
    format_memory_conflict_event,
    load_dialog_state,
    reset_dialog_state,
    save_dialog_state,
)
from memory_manager import load_session, reset_session, save_session


def evaluate_reset() -> tuple[bool, list[str]]:
    thread_id = "eval-session-controls-reset"
    save_session(
        thread_id,
        {
            "summary": "Apple discussion summary",
            "recent_messages": [{"role": "user", "content": "分析 Apple"}],
        },
    )
    save_dialog_state(
        thread_id,
        {
            "primary_company": "apple",
            "primary_ticker": "AAPL",
            "comparison_company": None,
            "comparison_ticker": None,
            "fiscal_year": 2024,
            "fiscal_quarter": None,
            "live_period": None,
            "live_interval": None,
            "analysis_focus": ["revenue"],
            "last_intent": "historical_rag",
            "last_query": "分析 Apple",
            "last_effective_query": "分析 Apple",
        },
    )
    failures: list[str] = []
    if not reset_session(thread_id):
        failures.append("reset_session returned False for existing session")
    if not reset_dialog_state(thread_id):
        failures.append("reset_dialog_state returned False for existing state")
    if load_session(thread_id).get("summary"):
        failures.append("adaptive_memory summary still exists after reset")
    if load_dialog_state(thread_id).get("primary_company"):
        failures.append("dialog_state primary_company still exists after reset")
    return not failures, failures


def evaluate_priority_note() -> tuple[bool, list[str], str]:
    conflicts = [
        "dialog_state.primary_company=nvidia missing from adaptive_memory companies=['apple']",
        "adaptive_memory contains stale company mentions not reflected in dialog_state: ['apple']",
    ]
    note = build_memory_priority_note(conflicts)
    failures: list[str] = []
    for token in [
        "prioritize the structured dialog state and resolved_query",
        "Detected memory/state conflicts",
        "primary_company=nvidia",
        "stale company mentions",
    ]:
        if token not in note:
            failures.append(f"missing token in priority note: {token}")
    return not failures, failures, note


def evaluate_conflict_event() -> tuple[bool, list[str], str]:
    event = build_memory_conflict_event(
        thread_id="demo-thread",
        conflicts=[
            "dialog_state.primary_company=nvidia missing from adaptive_memory companies=['apple']"
        ],
        dialog_state={
            "primary_company": "nvidia",
            "primary_ticker": "NVDA",
            "comparison_company": None,
            "comparison_ticker": None,
            "fiscal_year": 2025,
            "live_period": None,
            "analysis_focus": ["risk"],
            "last_intent": "historical_rag",
        },
        memory_session={
            "summary": "Apple remained the active company in summary.",
            "recent_messages": [{"role": "assistant", "content": "Apple context still appears."}],
        },
        effective_query="改看英伟达 2025 年利润和风险",
    )
    formatted = format_memory_conflict_event(event)
    failures: list[str] = []
    for token in [
        "[memory_event]",
        "\"event\": \"memory_state_conflict\"",
        "\"thread_id\": \"demo-thread\"",
        "\"conflict_count\": 1",
        "\"primary_company\": \"nvidia\"",
    ]:
        if token not in formatted:
            failures.append(f"missing token in formatted conflict event: {token}")
    return not failures, failures, formatted


def main() -> int:
    passed = 0

    ok, failures = evaluate_reset()
    print(f"[{'PASS' if ok else 'FAIL'}] 会话重置")
    if failures:
        print("  failures:")
        for item in failures:
            print(f"    - {item}")
    print()
    if ok:
        passed += 1

    ok, failures, note = evaluate_priority_note()
    print(f"[{'PASS' if ok else 'FAIL'}] 冲突优先级说明")
    print(f"  note: {note}")
    if failures:
        print("  failures:")
        for item in failures:
            print(f"    - {item}")
    print()
    if ok:
        passed += 1

    ok, failures, formatted = evaluate_conflict_event()
    print(f"[{'PASS' if ok else 'FAIL'}] 结构化冲突事件")
    print(f"  event: {formatted}")
    if failures:
        print("  failures:")
        for item in failures:
            print(f"    - {item}")
    print()
    if ok:
        passed += 1

    print(f"Summary: {passed}/3 passed")
    return 0 if passed == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
