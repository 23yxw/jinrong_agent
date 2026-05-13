"""Regression checks for market period extraction and MCP parameter defaults."""

from __future__ import annotations

import sys

from mcp_tools import _extract_market_params
from query_preprocessor import preprocess_query


CASES = [
    {
        "name": "中文 3 个月带空格",
        "query": "帮我看下 AAPL 最近 3 个月股价走势",
        "expected_ticker": "AAPL",
        "expected_period": "3mo",
        "expected_interval": "1d",
    },
    {
        "name": "中文 三个月",
        "query": "帮我看下特斯拉最近三个月股价走势",
        "expected_ticker": "TSLA",
        "expected_period": "3mo",
        "expected_interval": "1d",
    },
    {
        "name": "中文 1 周",
        "query": "看看 NVDA 最近 1 周走势",
        "expected_ticker": "NVDA",
        "expected_period": "1wk",
        "expected_interval": "1d",
    },
    {
        "name": "中文 6 个月",
        "query": "苹果过去 6 个月股价表现",
        "expected_ticker": "AAPL",
        "expected_period": "6mo",
        "expected_interval": "1d",
    },
    {
        "name": "中文 1 年",
        "query": "MSFT 过去 1 年走势",
        "expected_ticker": "MSFT",
        "expected_period": "1y",
        "expected_interval": "1d",
    },
    {
        "name": "中文 5 年",
        "query": "亚马逊最近 5 年股价趋势",
        "expected_ticker": "AMZN",
        "expected_period": "5y",
        "expected_interval": "1d",
    },
    {
        "name": "英文 last 3 months",
        "query": "AAPL stock price trend for the last 3 months",
        "expected_ticker": "AAPL",
        "expected_period": "3mo",
        "expected_interval": "1d",
    },
    {
        "name": "英文 past 6 months",
        "query": "Show me NVDA stock price trend for the past 6 months",
        "expected_ticker": "NVDA",
        "expected_period": "6mo",
        "expected_interval": "1d",
    },
    {
        "name": "英文 last one year",
        "query": "Show me TSLA stock price for the last one year",
        "expected_ticker": "TSLA",
        "expected_period": "1y",
        "expected_interval": "1d",
    },
    {
        "name": "period token 不应污染 interval",
        "query": "AAPL recent stock price 1mo",
        "expected_ticker": "AAPL",
        "expected_period": "1mo",
        "expected_interval": "1d",
    },
]


def evaluate_case(case: dict) -> tuple[bool, list[str], dict]:
    pre = preprocess_query(case["query"])
    ticker, period, interval = _extract_market_params(case["query"])
    failures: list[str] = []

    if pre.get("ticker") != case["expected_ticker"]:
        failures.append(f"pre.ticker={pre.get('ticker')}, expected {case['expected_ticker']}")
    if pre.get("live_period") != case["expected_period"]:
        failures.append(
            f"pre.live_period={pre.get('live_period')}, expected {case['expected_period']}"
        )
    if ticker != case["expected_ticker"]:
        failures.append(f"ticker={ticker}, expected {case['expected_ticker']}")
    if period != case["expected_period"]:
        failures.append(f"period={period}, expected {case['expected_period']}")
    if interval != case["expected_interval"]:
        failures.append(f"interval={interval}, expected {case['expected_interval']}")

    info = {
        "pre_ticker": pre.get("ticker"),
        "pre_live_period": pre.get("live_period"),
        "market_query_en": pre.get("market_query_en"),
        "ticker": ticker,
        "period": period,
        "interval": interval,
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
