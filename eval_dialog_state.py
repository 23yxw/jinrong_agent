"""Regression checks for structured dialog state inheritance across turns."""

from __future__ import annotations

import sys

from dialog_state_manager import (
    load_dialog_state,
    resolve_query_with_state,
    save_dialog_state,
    update_dialog_state,
)
from intent_router import route_query
from query_preprocessor import preprocess_query


def _blank_state() -> dict:
    return {
        "primary_company": None,
        "primary_ticker": None,
        "comparison_company": None,
        "comparison_ticker": None,
        "fiscal_year": None,
        "fiscal_quarter": None,
        "live_period": None,
        "live_interval": None,
        "analysis_focus": [],
        "last_intent": None,
        "last_query": None,
        "last_effective_query": None,
    }


def _prime_state(thread_id: str, query: str) -> None:
    save_dialog_state(thread_id, _blank_state())
    pre = preprocess_query(query)
    route = route_query(query, pre)
    update_dialog_state(
        thread_id,
        original_query=query,
        effective_query=query,
        preprocessed=pre,
        route=route,
    )


CASES = [
    {
        "name": "继承公司到行情追问",
        "thread_id": "eval-dialog-state-1",
        "prime_query": "分析一下特斯拉2024财报中的营收与利润趋势",
        "follow_query": "那最近1个月呢",
        "expected_company": "tesla",
        "expected_ticker": "TSLA",
        "expected_live_period": "1mo",
        "expect_state_used": True,
    },
    {
        "name": "继承基础公司做比较",
        "thread_id": "eval-dialog-state-2",
        "prime_query": "分析一下 Apple 2024 revenue growth",
        "follow_query": "和微软比一下",
        "expected_company": "apple",
        "expected_ticker": "AAPL",
        "expected_effective_contains": ["compare apple with microsoft"],
        "expect_state_used": True,
    },
    {
        "name": "继承年份和公司到维度追问",
        "thread_id": "eval-dialog-state-3",
        "prime_query": "总结 Microsoft 2024 revenue 和 cash flow",
        "follow_query": "只看现金流",
        "expected_company": "microsoft",
        "expected_ticker": "MSFT",
        "expected_fiscal_year": 2024,
        "expect_state_used": True,
    },
    {
        "name": "切换公司覆盖上一轮主语",
        "thread_id": "eval-dialog-state-4",
        "prime_query": "分析 Apple 2024 财报",
        "follow_query": "改看英伟达 2025 年利润和风险",
        "expected_company": "nvidia",
        "expected_ticker": "NVDA",
        "expected_fiscal_year": 2025,
        "expect_state_used": False,
        "expected_post_state_company": "nvidia",
    },
    {
        "name": "切换年份覆盖上一轮年份",
        "thread_id": "eval-dialog-state-5",
        "prime_query": "分析 Tesla 2024 revenue growth",
        "follow_query": "那 2023 年呢",
        "expected_company": "tesla",
        "expected_ticker": "TSLA",
        "expected_fiscal_year": 2023,
        "expect_state_used": True,
        "expected_post_state_year": 2023,
    },
    {
        "name": "取消比较清理 comparison 状态",
        "thread_id": "eval-dialog-state-6",
        "prime_query": "把 Apple 和 Microsoft 的盈利能力对比一下",
        "follow_query": "先不比较了，只看苹果利润率",
        "expected_company": "apple",
        "expected_ticker": "AAPL",
        "expect_state_used": False,
        "expected_post_state_company": "apple",
        "expect_comparison_cleared": True,
    },
    {
        "name": "反问式追问继承上一轮公司",
        "thread_id": "eval-dialog-state-7",
        "prime_query": "总结 Amazon 2024 cash flow and margin",
        "follow_query": "风险大吗？",
        "expected_company": "amazon",
        "expected_ticker": "AMZN",
        "expected_fiscal_year": 2024,
        "expect_state_used": True,
        "expected_trace_reasons": [
            "inherit_primary_company_for_follow_up",
            "inherit_fiscal_year_for_document_follow_up",
        ],
    },
]


def evaluate_case(case: dict) -> tuple[bool, list[str], dict]:
    _prime_state(case["thread_id"], case["prime_query"])
    resolution = resolve_query_with_state(case["thread_id"], case["follow_query"])
    pre = resolution["preprocessed"]
    trace = resolution.get("dialog_state_trace") or {}
    failures = []

    if resolution.get("state_used") != case.get("expect_state_used"):
        failures.append(
            f"state_used={resolution.get('state_used')}, expected {case.get('expect_state_used')}"
        )
    if pre.get("canonical_company") != case.get("expected_company"):
        failures.append(
            f"company={pre.get('canonical_company')}, expected {case.get('expected_company')}"
        )
    if pre.get("ticker") != case.get("expected_ticker"):
        failures.append(f"ticker={pre.get('ticker')}, expected {case.get('expected_ticker')}")
    if case.get("expected_live_period") and pre.get("live_period") != case["expected_live_period"]:
        failures.append(
            f"live_period={pre.get('live_period')}, expected {case['expected_live_period']}"
        )
    if case.get("expected_fiscal_year") and pre.get("fiscal_year") != case["expected_fiscal_year"]:
        failures.append(
            f"fiscal_year={pre.get('fiscal_year')}, expected {case['expected_fiscal_year']}"
        )
    for token in case.get("expected_effective_contains", []):
        if token.lower() not in (resolution.get("effective_query") or "").lower():
            failures.append(f"effective_query missing token: {token}")
    for reason in case.get("expected_trace_reasons", []):
        if reason not in (trace.get("reasons") or []):
            failures.append(f"trace missing reason: {reason}")

    route = route_query(resolution["effective_query"], pre)
    update_dialog_state(
        case["thread_id"],
        original_query=case["follow_query"],
        effective_query=resolution["effective_query"],
        preprocessed=pre,
        route=route,
    )
    post_state = load_dialog_state(case["thread_id"])
    if case.get("expected_post_state_company") and post_state.get("primary_company") != case["expected_post_state_company"]:
        failures.append(
            f"post_state.primary_company={post_state.get('primary_company')}, "
            f"expected {case['expected_post_state_company']}"
        )
    if case.get("expected_post_state_year") and post_state.get("fiscal_year") != case["expected_post_state_year"]:
        failures.append(
            f"post_state.fiscal_year={post_state.get('fiscal_year')}, expected {case['expected_post_state_year']}"
        )
    if case.get("expect_comparison_cleared") and (
        post_state.get("comparison_company") or post_state.get("comparison_ticker")
    ):
        failures.append(
            "comparison state not cleared: "
            f"{post_state.get('comparison_company')}/{post_state.get('comparison_ticker')}"
        )

    info = {
        "effective_query": resolution.get("effective_query"),
        "state_used": resolution.get("state_used"),
        "company": pre.get("canonical_company"),
        "ticker": pre.get("ticker"),
        "fiscal_year": pre.get("fiscal_year"),
        "live_period": pre.get("live_period"),
        "trace_reasons": trace.get("reasons"),
        "post_state_company": post_state.get("primary_company"),
        "post_state_comparison": post_state.get("comparison_company"),
        "post_state_year": post_state.get("fiscal_year"),
    }
    return not failures, failures, info


def main() -> int:
    passed = 0
    for case in CASES:
        ok, failures, info = evaluate_case(case)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"  follow_query: {case['follow_query']}")
        print(f"  info: {info}")
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
