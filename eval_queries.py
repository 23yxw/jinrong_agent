"""Lightweight regression checks for deterministic preprocessing and semantic retrieval rewrites."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys

from intent_router import route_query
from query_preprocessor import preprocess_query
from retrieval import rewrite_query_for_rag


@dataclass
class QueryCase:
    name: str
    query: str
    allowed_intents: set[str]
    expected_company: str | None = None
    expected_ticker: str | None = None
    expected_fiscal_year: int | None = None
    expected_needs_clarification: bool | None = None
    expected_tools: set[str] | None = None
    require_report_query: bool = False
    require_market_query: bool = False
    require_mixed_language: bool | None = None
    retrieval_must_include_any: list[str] = field(default_factory=list)
    retrieval_must_include_all: list[str] = field(default_factory=list)
    retrieval_min_queries: int = 0
    expect_empty_retrieval: bool = False


@dataclass
class CaseResult:
    case: QueryCase
    ok: bool
    failures: list[str] = field(default_factory=list)
    route: dict = field(default_factory=dict)
    preprocessed: dict = field(default_factory=dict)
    retrieval_queries: list[str] = field(default_factory=list)


CASES = [
    QueryCase(
        name="中文财报分析紧贴年份",
        query="分析一下苹果2024财报中的营收与利润趋势",
        allowed_intents={"historical_rag"},
        expected_company="apple",
        expected_ticker="AAPL",
        expected_fiscal_year=2024,
        expected_needs_clarification=False,
        expected_tools={"deep_rag_search"},
        require_report_query=True,
        require_mixed_language=False,
        retrieval_must_include_all=["apple", "2024"],
        retrieval_must_include_any=["revenue", "profit", "10-k"],
        retrieval_min_queries=2,
    ),
    QueryCase(
        name="中文财报分析",
        query="苹果 2024 财报里的营收和净利润趋势怎么样？",
        allowed_intents={"historical_rag"},
        expected_company="apple",
        expected_ticker="AAPL",
        expected_needs_clarification=False,
        expected_tools={"deep_rag_search"},
        require_report_query=True,
        require_mixed_language=False,
        retrieval_must_include_all=["apple", "2024"],
        retrieval_must_include_any=["revenue", "net income", "profit", "10-k"],
        retrieval_min_queries=2,
    ),
    QueryCase(
        name="中文行情查询",
        query="帮我看下 AAPL 最近 3 个月股价走势",
        allowed_intents={"live_market"},
        expected_company="apple",
        expected_ticker="AAPL",
        expected_needs_clarification=False,
        expected_tools={"live_finance_researcher"},
        require_market_query=True,
        require_mixed_language=True,
        expect_empty_retrieval=True,
    ),
    QueryCase(
        name="中英混杂财报问题",
        query="Apple 2024 revenue risk 和 guidance 怎么看",
        allowed_intents={"historical_rag", "finance_general"},
        expected_company="apple",
        expected_ticker="AAPL",
        expected_needs_clarification=False,
        expected_tools={"deep_rag_search"},
        require_report_query=True,
        require_mixed_language=True,
        retrieval_must_include_all=["apple", "2024"],
        retrieval_must_include_any=["revenue", "risk", "guidance", "10-k"],
        retrieval_min_queries=2,
    ),
    QueryCase(
        name="混合历史与实时",
        query="我想看看特斯拉，是财报表现更重要还是最近股价更重要？",
        allowed_intents={"hybrid"},
        expected_company="tesla",
        expected_ticker="TSLA",
        expected_needs_clarification=False,
        expected_tools={"deep_rag_search", "live_finance_researcher"},
        require_report_query=True,
        require_market_query=True,
        require_mixed_language=False,
        retrieval_must_include_all=["tesla"],
        retrieval_must_include_any=["stock price", "financial report", "10-k"],
        retrieval_min_queries=2,
    ),
    QueryCase(
        name="英文实时行情",
        query="Show me NVDA stock price trend for the last 6 months",
        allowed_intents={"live_market"},
        expected_company="nvidia",
        expected_ticker="NVDA",
        expected_needs_clarification=False,
        expected_tools={"live_finance_researcher"},
        require_market_query=True,
        require_mixed_language=False,
        expect_empty_retrieval=True,
    ),
    QueryCase(
        name="英文财务指标",
        query="Summarize Microsoft's 2024 revenue, margin, and cash flow",
        allowed_intents={"historical_rag", "finance_general"},
        expected_company="microsoft",
        expected_ticker="MSFT",
        expected_needs_clarification=False,
        expected_tools={"deep_rag_search"},
        require_report_query=True,
        require_mixed_language=False,
        retrieval_must_include_all=["microsoft", "2024"],
        retrieval_must_include_any=["revenue", "margin", "cash flow"],
        retrieval_min_queries=2,
    ),
    QueryCase(
        name="非金融问题",
        query="今天天气怎么样",
        allowed_intents={"finance_general", "non_finance"},
        expected_needs_clarification=True,
        expected_tools=set(),
        require_mixed_language=False,
        expect_empty_retrieval=True,
    ),
]


def evaluate_case(case: QueryCase, *, include_retrieval: bool = False) -> CaseResult:
    pre = preprocess_query(case.query)
    route = route_query(case.query, pre)
    failures: list[str] = []
    retrieval_queries: list[str] = []

    intent = route.get("intent")
    if intent not in case.allowed_intents:
        failures.append(f"intent={intent}, expected one of {sorted(case.allowed_intents)}")

    if case.expected_company is not None and pre.get("canonical_company") != case.expected_company:
        failures.append(
            f"company={pre.get('canonical_company')}, expected {case.expected_company}"
        )

    if case.expected_ticker is not None and pre.get("ticker") != case.expected_ticker:
        failures.append(f"ticker={pre.get('ticker')}, expected {case.expected_ticker}")

    if case.expected_fiscal_year is not None and pre.get("fiscal_year") != case.expected_fiscal_year:
        failures.append(
            f"fiscal_year={pre.get('fiscal_year')}, expected {case.expected_fiscal_year}"
        )

    if (
        case.expected_needs_clarification is not None
        and route.get("needs_clarification") != case.expected_needs_clarification
    ):
        failures.append(
            "needs_clarification="
            f"{route.get('needs_clarification')}, expected {case.expected_needs_clarification}"
        )

    if case.expected_tools is not None:
        actual_tools = set(route.get("recommended_tools") or [])
        if actual_tools != case.expected_tools:
            failures.append(f"tools={sorted(actual_tools)}, expected {sorted(case.expected_tools)}")

    if case.require_report_query and not pre.get("report_query_en"):
        failures.append("report_query_en is empty")

    if case.require_market_query and not pre.get("market_query_en"):
        failures.append("market_query_en is empty")

    if (
        case.require_mixed_language is not None
        and bool(pre.get("is_mixed_language")) != case.require_mixed_language
    ):
        failures.append(
            f"is_mixed_language={pre.get('is_mixed_language')}, "
            f"expected {case.require_mixed_language}"
        )

    if include_retrieval:
        retrieval_queries = rewrite_query_for_rag(case.query)
        _validate_retrieval_queries(case, retrieval_queries, failures)

    return CaseResult(
        case=case,
        ok=not failures,
        failures=failures,
        route=route,
        preprocessed=pre,
        retrieval_queries=retrieval_queries,
    )


def format_result(result: CaseResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    case = result.case
    pre = result.preprocessed
    route = result.route
    lines = [
        f"[{status}] {case.name}",
        f"  query: {case.query}",
        (
            "  route: "
            f"intent={route.get('intent')} "
            f"rule={route.get('rule_intent')} "
            f"semantic={route.get('semantic_intent')} "
            f"clarify={route.get('needs_clarification')} "
            f"tools={route.get('recommended_tools')}"
        ),
        (
            "  pre: "
            f"company={pre.get('canonical_company')} "
            f"ticker={pre.get('ticker')} "
            f"mixed={pre.get('is_mixed_language')} "
            f"report_en={bool(pre.get('report_query_en'))} "
            f"market_en={bool(pre.get('market_query_en'))}"
        ),
    ]
    if result.retrieval_queries:
        lines.append(f"  retrieval: {result.retrieval_queries[:4]}")
    if result.failures:
        lines.append("  failures:")
        for item in result.failures:
            lines.append(f"    - {item}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate bilingual finance query behavior.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failing query case.",
    )
    parser.add_argument(
        "--mode",
        choices=["routing", "retrieval", "all"],
        default="all",
        help="Choose which layer to validate.",
    )
    args = parser.parse_args()

    results: list[CaseResult] = []
    include_retrieval = args.mode in {"retrieval", "all"}
    for case in CASES:
        result = evaluate_case(case, include_retrieval=include_retrieval)
        results.append(result)
        print(format_result(result))
        print()
        if args.fail_fast and not result.ok:
            break

    failed = [item for item in results if not item.ok]
    passed_count = len(results) - len(failed)
    print(f"Summary: {passed_count}/{len(results)} passed")
    if failed:
        print("Failed cases:")
        for item in failed:
            print(f"- {item.case.name}")
        return 1
    return 0


def _validate_retrieval_queries(
    case: QueryCase,
    retrieval_queries: list[str],
    failures: list[str],
) -> None:
    if case.expect_empty_retrieval:
        if retrieval_queries:
            failures.append(f"retrieval queries should be empty, got {retrieval_queries[:4]}")
        return

    if len(retrieval_queries) < case.retrieval_min_queries:
        failures.append(
            f"retrieval_queries={len(retrieval_queries)}, expected at least {case.retrieval_min_queries}"
        )
        return

    joined = " | ".join(item.lower() for item in retrieval_queries)
    for token in case.retrieval_must_include_all:
        if token.lower() not in joined:
            failures.append(f"retrieval queries missing required token: {token}")

    if case.retrieval_must_include_any:
        if not any(token.lower() in joined for token in case.retrieval_must_include_any):
            failures.append(
                "retrieval queries missing any of: "
                + ", ".join(case.retrieval_must_include_any)
            )


if __name__ == "__main__":
    sys.exit(main())
