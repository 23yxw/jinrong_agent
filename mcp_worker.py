"""Child-process worker for isolated MCP tool execution."""

from __future__ import annotations

import argparse
import asyncio
import json


def _to_text(raw) -> str:
    data = raw.content if hasattr(raw, "content") else raw
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(data)


def _pick_mcp_tools_by_intent(tools: list, intent: str) -> list:
    alias_map = {
        "historical": ["historical", "price_history", "stock_prices", "get_historical_stock_prices"],
        "quote": ["quote", "current_price", "stock_summary", "ticker", "get_stock_price"],
        "news": ["news", "headline", "get_stock_news"],
        "analyst": ["analyst", "recommendation", "rating"],
        "fundamental": ["fundamental", "financial", "income_statement", "balance_sheet", "cash_flow", "market_cap"],
    }
    aliases = alias_map.get(intent, [])
    scored = []
    for tool in tools:
        name = tool.name.lower()
        score = 0
        for alias in aliases:
            if alias in name:
                score += 2
        if "stock" in name:
            score += 1
        if "yahoo" in name:
            score += 1
        if score > 0:
            scored.append((score, tool))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tool for _, tool in scored]


def _build_tool_input(tool, query: str, ticker: str, period: str, interval: str) -> dict:
    payload = {
        "query": query,
        "ticker": ticker,
        "symbol": ticker,
        "period": period,
        "interval": interval,
    }
    accepted = None
    try:
        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            accepted = set(args.keys())
    except Exception:  # noqa: BLE001
        accepted = None
    if not accepted:
        return payload
    filtered = {k: v for k, v in payload.items() if k in accepted}
    return filtered or payload


async def _run(payload: dict) -> dict:
    try:
        from langchain_mcp_adapters import MultiServerMCPClient
    except ImportError:
        from langchain_mcp_adapters.client import MultiServerMCPClient

    query = payload.get("query", "")
    ticker = payload.get("ticker", "TSLA")
    period = payload.get("period", "1mo")
    interval = payload.get("interval", "1d")
    intent = payload.get("intent", "quote")

    client = MultiServerMCPClient(
        {
            "yahoo_finance": {
                "command": "uvx",
                "args": ["yahoo-finance-mcp-server"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    candidates = _pick_mcp_tools_by_intent(tools, intent)
    if not candidates:
        return {
            "ok": False,
            "source": "mcp",
            "error": f"no_candidate_tools intent={intent}",
            "tools": [t.name for t in tools],
        }

    last_error = None
    for tool in candidates:
        tool_input = _build_tool_input(tool, query, ticker, period, interval)
        try:
            try:
                raw = await tool.ainvoke(tool_input)
            except TypeError:
                raw = await tool.ainvoke(**tool_input)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{tool.name}: {type(exc).__name__}: {repr(exc)}"
            continue

        return {
            "ok": True,
            "source": "mcp",
            "tool_name": tool.name,
            "intent": intent,
            "result": raw.content if hasattr(raw, "content") else raw,
            "result_text": _to_text(raw),
            "params": {
                "ticker": ticker,
                "period": period,
                "interval": interval,
            },
        }

    return {
        "ok": False,
        "source": "mcp",
        "error": f"all_tools_failed: {last_error}",
    }


def main():
    parser = argparse.ArgumentParser(description="Run one isolated MCP task.")
    parser.add_argument("--payload", required=True, help="JSON payload")
    args = parser.parse_args()
    payload = json.loads(args.payload)
    result = asyncio.run(_run(payload))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
