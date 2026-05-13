"""Regression checks for hybrid query splitting and routing notes."""

from __future__ import annotations

import sys

from intent_router import build_agent_routing_note, route_query
from query_preprocessor import (
    HYBRID_DOCUMENT_AVOID_TERMS,
    HYBRID_MARKET_AVOID_TERMS,
    preprocess_query,
    semantic_enrich_query,
    split_hybrid_query,
)


CASES = [
    {
        "name": "Tesla 混合问题",
        "query": "我想看看特斯拉，是财报表现更重要还是最近股价更重要？",
        "expected_company": "tesla",
        "expected_ticker": "TSLA",
        "document_must_include_any": ["tesla", "revenue", "profit", "margin", "cash flow"],
        "market_must_include_any": ["tsla", "stock price", "1mo"],
    },
    {
        "name": "Apple 混合问题",
        "query": "Apple 2024 revenue growth 和最近股价走势一起看",
        "expected_company": "apple",
        "expected_ticker": "AAPL",
        "document_must_include_any": ["apple", "2024", "revenue"],
        "market_must_include_any": ["aapl", "stock price", "trend"],
    },
]


def evaluate_case(case: dict) -> tuple[bool, list[str], dict]:
    query = case["query"]
    pre = preprocess_query(query)
    route = route_query(query, pre)
    semantic = semantic_enrich_query(query, pre)
    split = split_hybrid_query(query, pre, semantic)
    note = build_agent_routing_note(query, preprocessed=pre, route=route)

    failures = []
    if route.get("intent") != "hybrid":
        failures.append(f"intent={route.get('intent')}, expected hybrid")
    if pre.get("canonical_company") != case["expected_company"]:
        failures.append(
            f"company={pre.get('canonical_company')}, expected {case['expected_company']}"
        )
    if pre.get("ticker") != case["expected_ticker"]:
        failures.append(f"ticker={pre.get('ticker')}, expected {case['expected_ticker']}")
    if not split.get("document_query"):
        failures.append("document_query is empty")
    if not split.get("market_query"):
        failures.append("market_query is empty")
    if "document_query=" not in note:
        failures.append("routing note missing document_query")
    if "market_query=" not in note:
        failures.append("routing note missing market_query")

    document_query = (split.get("document_query") or "").lower()
    market_query = (split.get("market_query") or "").lower()
    for token in HYBRID_DOCUMENT_AVOID_TERMS:
        if token in document_query:
            failures.append(f"document_query contains market-only term: {token}")
    for token in HYBRID_MARKET_AVOID_TERMS:
        if token in market_query:
            failures.append(f"market_query contains filing-only term: {token}")

    doc_any = case.get("document_must_include_any", [])
    if doc_any and not any(token.lower() in document_query for token in doc_any):
        failures.append(f"document_query missing any of {doc_any}")
    market_any = case.get("market_must_include_any", [])
    if market_any and not any(token.lower() in market_query for token in market_any):
        failures.append(f"market_query missing any of {market_any}")

    info = {
        "route_intent": route.get("intent"),
        "semantic_intent": route.get("semantic_intent"),
        "document_query": split.get("document_query"),
        "market_query": split.get("market_query"),
        "analysis_source": split.get("analysis_source"),
    }
    return not failures, failures, info


def main() -> int:
    passed = 0
    for case in CASES:
        ok, failures, info = evaluate_case(case)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"  query: {case['query']}")
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
