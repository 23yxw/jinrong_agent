"""Checks consistency between adaptive_memory and structured dialog_state."""

from __future__ import annotations

import sys

from dialog_state_manager import detect_memory_state_conflicts, save_dialog_state
from memory_manager import load_session, save_session


def _session(summary: str, recent_messages: list[dict]) -> dict:
    return {
        "summary": summary,
        "recent_messages": recent_messages,
    }


CASES = [
    {
        "name": "一致的 Apple 上下文",
        "thread_id": "eval-memory-consistency-1",
        "session": _session(
            summary="User is analyzing Apple 2024 revenue and cash flow. Stock discussion uses AAPL and 1mo trend.",
            recent_messages=[
                {"role": "user", "content": "分析 Apple 2024 revenue"},
                {"role": "assistant", "content": "Apple (AAPL) 2024 revenue is discussed with 1mo stock context."},
            ],
        ),
        "state": {
            "primary_company": "apple",
            "primary_ticker": "AAPL",
            "comparison_company": None,
            "comparison_ticker": None,
            "fiscal_year": 2024,
            "fiscal_quarter": None,
            "live_period": "1mo",
            "live_interval": "1d",
            "analysis_focus": ["revenue", "cash flow"],
            "last_intent": "hybrid",
            "last_query": "分析 Apple 2024 revenue",
            "last_effective_query": "分析 Apple 2024 revenue",
        },
        "expected_conflict_count": 0,
    },
    {
        "name": "摘要公司与状态主公司冲突",
        "thread_id": "eval-memory-consistency-2",
        "session": _session(
            summary="Conversation mainly discusses Tesla 2024 margins and risk.",
            recent_messages=[{"role": "assistant", "content": "Tesla remains the active subject."}],
        ),
        "state": {
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
        "expected_conflict_substrings": ["primary_company=apple"],
    },
    {
        "name": "ticker 缺失冲突",
        "thread_id": "eval-memory-consistency-3",
        "session": _session(
            summary="Apple is the main company. The discussion focuses on stock momentum.",
            recent_messages=[{"role": "assistant", "content": "Apple stock is being watched closely."}],
        ),
        "state": {
            "primary_company": "apple",
            "primary_ticker": "AAPL",
            "comparison_company": None,
            "comparison_ticker": None,
            "fiscal_year": None,
            "fiscal_quarter": None,
            "live_period": None,
            "live_interval": None,
            "analysis_focus": ["stock price"],
            "last_intent": "live_market",
            "last_query": "看 Apple 股价",
            "last_effective_query": "看 Apple 股价",
        },
        "expected_conflict_substrings": ["primary_ticker=AAPL"],
    },
    {
        "name": "比较对象缺失冲突",
        "thread_id": "eval-memory-consistency-4",
        "session": _session(
            summary="The user compares Apple margins and revenue trends.",
            recent_messages=[{"role": "assistant", "content": "Apple is compared against prior-year performance."}],
        ),
        "state": {
            "primary_company": "apple",
            "primary_ticker": "AAPL",
            "comparison_company": "microsoft",
            "comparison_ticker": "MSFT",
            "fiscal_year": 2024,
            "fiscal_quarter": None,
            "live_period": None,
            "live_interval": None,
            "analysis_focus": ["margin"],
            "last_intent": "historical_rag",
            "last_query": "Apple 和 Microsoft 对比",
            "last_effective_query": "Apple 和 Microsoft 对比",
        },
        "expected_conflict_substrings": ["comparison_company=microsoft"],
    },
    {
        "name": "摘要残留旧公司但状态已切换新公司",
        "thread_id": "eval-memory-consistency-5",
        "session": _session(
            summary="Earlier the conversation focused on Apple revenue and Apple margin trends.",
            recent_messages=[
                {"role": "assistant", "content": "Apple remained the focus in the compressed history."},
            ],
        ),
        "state": {
            "primary_company": "nvidia",
            "primary_ticker": "NVDA",
            "comparison_company": None,
            "comparison_ticker": None,
            "fiscal_year": 2025,
            "fiscal_quarter": None,
            "live_period": None,
            "live_interval": None,
            "analysis_focus": ["profit", "risk"],
            "last_intent": "historical_rag",
            "last_query": "改看英伟达 2025 年利润和风险",
            "last_effective_query": "改看英伟达 2025 年利润和风险",
        },
        "expected_conflict_substrings": [
            "primary_company=nvidia",
            "stale company mentions",
        ],
    },
    {
        "name": "摘要同时残留旧公司和新公司但无比较状态",
        "thread_id": "eval-memory-consistency-6",
        "session": _session(
            summary="The recent discussion moved to NVIDIA, but Apple revenue is still repeatedly mentioned in history.",
            recent_messages=[
                {"role": "assistant", "content": "NVIDIA is now the active company, while Apple still appears in prior notes."},
            ],
        ),
        "state": {
            "primary_company": "nvidia",
            "primary_ticker": "NVDA",
            "comparison_company": None,
            "comparison_ticker": None,
            "fiscal_year": 2025,
            "fiscal_quarter": None,
            "live_period": None,
            "live_interval": None,
            "analysis_focus": ["revenue"],
            "last_intent": "historical_rag",
            "last_query": "看英伟达收入",
            "last_effective_query": "看英伟达收入",
        },
        "expected_conflict_substrings": ["stale company mentions"],
    },
]


def evaluate_case(case: dict) -> tuple[bool, list[str], list[str]]:
    save_session(case["thread_id"], case["session"])
    save_dialog_state(case["thread_id"], case["state"])
    session = load_session(case["thread_id"])
    conflicts = detect_memory_state_conflicts(session, case["state"])
    failures: list[str] = []

    expected_count = case.get("expected_conflict_count")
    if expected_count is not None and len(conflicts) != expected_count:
        failures.append(f"conflict_count={len(conflicts)}, expected {expected_count}")

    for token in case.get("expected_conflict_substrings", []):
        if not any(token in item for item in conflicts):
            failures.append(f"missing expected conflict token: {token}")

    return not failures, failures, conflicts


def main() -> int:
    passed = 0
    for case in CASES:
        ok, failures, conflicts = evaluate_case(case)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"  conflicts: {conflicts}")
        if failures:
            print("  failures:")
            for item in failures:
                print(f"    - {item}")
        print()
        if ok:
            passed += 1
    print(f"Summary: {passed}/{len(CASES)} passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
