"""Yahoo Finance MCP-related tools and wrappers."""

import json
import subprocess
import sys

from query_preprocessor import preprocess_query
from runtime_checks import assert_runtime_ready
from sandbox_scheduler import run_mcp_task

MCP_CALL_TIMEOUT_SECONDS = 90


async def get_tools():
    assert_runtime_ready(
        stage="mcp_tools.get_tools",
        packages=["langchain_mcp_adapters", "langchain"],
        commands=["uvx"],
    )
    try:
        from langchain_mcp_adapters import MultiServerMCPClient
    except ImportError:
        # Compatibility with newer langchain_mcp_adapters versions.
        from langchain_mcp_adapters.client import MultiServerMCPClient

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
    return tools


async def finance_research(query: str) -> str:
    # Keep compatibility for old callers, but remove the inner LLM-agent layer.
    return await _live_finance_research_via_mcp(query)


def finance_researcher(query: str) -> str:
    """Research stocks using Yahoo Finance MCP async function."""
    safe_query = repr(query)
    code = f"""
import asyncio
from mcp_tools import finance_research
result = asyncio.run(finance_research({safe_query}))
print(result)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=MCP_CALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: MCP call timed out after {MCP_CALL_TIMEOUT_SECONDS}s. "
            "Please retry or check MCP server health."
        )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout.strip()


def live_finance_researcher(query: str) -> str:
    """Research live stock data using Yahoo Finance MCP."""
    if not query or not query.strip():
        return "Empty query provided."

    mcp_error = None
    try:
        output = _live_finance_research_via_sandbox(query)
        if output and _mcp_output_usable(output):
            if len(output) > 6000:
                output = output[:6000] + "\n\n...[truncated]"
            return output
        mcp_error = f"MCP returned empty or unusable data. Raw output: {repr(output)[:500]}"
    except TimeoutError as exc:
        mcp_error = (
            f"MCP timeout after {MCP_CALL_TIMEOUT_SECONDS}s "
            f"({type(exc).__name__}: {repr(exc)})"
        )
    except Exception as exc:  # noqa: BLE001
        mcp_error = f"MCP exception ({type(exc).__name__}): {repr(exc)}"

    fallback = _fallback_live_finance_by_yfinance(query, reason=mcp_error)
    if len(fallback) > 6000:
        fallback = fallback[:6000] + "\n\n...[truncated]"
    return fallback


def _live_finance_research_via_sandbox(query: str) -> str:
    ticker, period, interval = _extract_market_params(query)
    intent = _classify_live_query_intent(query)
    payload = {
        "query": query,
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "intent": intent,
    }
    result = run_mcp_task(payload, timeout_seconds=MCP_CALL_TIMEOUT_SECONDS)
    if not result.get("ok"):
        return f"MCP error: {result.get('error')}"

    tool_name = result.get("tool_name", "unknown_tool")
    raw = result.get("result")
    if intent == "historical" or _tool_name_is_historical(tool_name):
        summarized = _summarize_mcp_historical_output(raw, ticker, period, interval)
        if summarized and _mcp_output_usable(summarized):
            return f"Source: MCP(yfinance:{tool_name})\n" + summarized.replace(
                "Source: MCP(yfinance)\n", "", 1
            )
        return summarized
    text = result.get("result_text") or _to_text(raw)
    return f"Source: MCP(yfinance:{tool_name})\n{text}"


async def _live_finance_research_via_mcp(query: str) -> str:
    """Directly call MCP stock tools and summarize results without inner LLM agent."""
    ticker, period, interval = _extract_market_params(query)
    tools = await get_tools()
    tool_names = [t.name for t in tools]

    intent = _classify_live_query_intent(query)
    candidates = _pick_mcp_tools_by_intent(tools, intent)
    if not candidates:
        return f"MCP error: no suitable tool found. Available tools: {tool_names}"

    last_error = None
    for target in candidates:
        tool_input = _build_mcp_tool_input(target, query, ticker, period, interval)
        try:
            try:
                raw = await target.ainvoke(tool_input)
            except TypeError:
                raw = await target.ainvoke(**tool_input)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{target.name}: {type(exc).__name__}: {repr(exc)}"
            continue

        if intent == "historical":
            summarized = _summarize_mcp_historical_output(raw, ticker, period, interval)
            if _mcp_output_usable(summarized):
                return summarized
        else:
            text = _to_text(raw)
            if _mcp_output_usable(text):
                return f"Source: MCP(yfinance:{target.name})\n{text}"
        last_error = f"{target.name}: unusable output"

    return f"MCP error: all candidate tools failed. Last error: {last_error}"


def _classify_live_query_intent(query: str) -> str:
    text = query.lower()
    if any(key in text for key in ["news", "新闻", "headline", "头条"]):
        return "news"
    if any(key in text for key in ["analyst", "rating", "recommendation", "评级", "研报"]):
        return "analyst"
    if any(key in text for key in ["market cap", "估值", "市值", "pe", "财务", "fundamental"]):
        return "fundamental"
    if any(key in text for key in ["current", "latest", "today", "现价", "当前", "最新"]) and not any(
        key in text for key in ["trend", "走势", "历史", "recent", "最近", "past", "last", "month", "week", "year", "月", "周", "年"]
    ):
        return "quote"
    if any(
        key in text
        for key in [
            "trend",
            "走势",
            "历史",
            "period",
            "interval",
            "recent",
            "最近",
            "past",
            "last",
            "month",
            "months",
            "week",
            "weeks",
            "year",
            "years",
            "月",
            "周",
            "年",
            "日线",
            "stock price",
        ]
    ):
        return "historical"
    return "quote"


def _pick_mcp_tools_by_intent(tools: list, intent: str) -> list:
    alias_map = {
        "historical": [
            "get_historical_stock_prices",
            "historical",
            "price_history",
            "stock_prices",
        ],
        "quote": [
            "get_stock_price",
            "quote",
            "current_price",
            "stock_summary",
            "ticker",
        ],
        "news": [
            "news",
            "headline",
            "get_stock_news",
        ],
        "analyst": [
            "analyst",
            "recommendation",
            "rating",
        ],
        "fundamental": [
            "fundamental",
            "financial",
            "income_statement",
            "balance_sheet",
            "cash_flow",
            "market_cap",
        ],
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


def _build_mcp_tool_input(tool, query: str, ticker: str, period: str, interval: str) -> dict:
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


def _to_text(raw) -> str:
    data = raw.content if hasattr(raw, "content") else raw
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(data)


def _summarize_mcp_historical_output(raw, ticker: str, period: str, interval: str) -> str:
    data = raw.content if hasattr(raw, "content") else raw
    data = _unwrap_mcp_payload(data)
    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            return ""
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                data = json.loads(stripped)
            except Exception:  # noqa: BLE001
                return data
        else:
            return data

    closes = _extract_close_series(data)
    if len(closes) < 2:
        return f"MCP returned non-tabular or insufficient price series: {str(data)[:600]}"

    start_price = closes[0]
    end_price = closes[-1]
    change_pct = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
    high = max(closes)
    low = min(closes)
    trend = "uptrend" if change_pct > 1 else "downtrend" if change_pct < -1 else "sideways"
    return (
        "Source: MCP(yfinance)\n"
        f"- Ticker: {ticker}\n"
        f"- Period/Interval: {period} / {interval}\n"
        f"- Start Close: {start_price:.2f}\n"
        f"- End Close: {end_price:.2f}\n"
        f"- Change: {change_pct:.2f}%\n"
        f"- High/Low: {high:.2f} / {low:.2f}\n"
        f"- Trend: {trend}\n"
    )


def _unwrap_mcp_payload(data):
    """Normalize common MCP content wrappers into plain JSON-like payload."""
    if isinstance(data, list):
        # Common MCP format: [{"type":"text","text":"[...]"}]
        if all(isinstance(item, dict) and "text" in item for item in data):
            combined = "\n".join(str(item.get("text", "")) for item in data).strip()
            if combined.startswith("{") or combined.startswith("["):
                try:
                    return json.loads(combined)
                except Exception:  # noqa: BLE001
                    return combined
            return combined
    return data


def _extract_close_series(data) -> list[float]:
    if isinstance(data, dict):
        for key in ["prices", "data", "result", "historical_prices", "items", "rows"]:
            if key in data:
                return _extract_close_series(data[key])
        return []

    if isinstance(data, list):
        values = []
        for item in data:
            if not isinstance(item, dict):
                continue
            close = item.get("close", item.get("Close"))
            if close is None:
                continue
            try:
                values.append(float(close))
            except Exception:  # noqa: BLE001
                continue
        return values

    return []


def _mcp_output_usable(output: str) -> bool:
    text = output.strip().lower()
    if not text:
        return False
    bad_signals = [
        "无法获取",
        "无法提供",
        "no live finance data found",
        "error",
        "tool error",
        "empty",
        "data is empty",
        "cannot fulfill",
        "i am sorry",
        "lack the ability",
        "cannot provide",
        "unable to",
        "non-tabular or insufficient price series",
    ]
    return not any(signal in text for signal in bad_signals)


def _extract_market_params(query: str) -> tuple[str, str, str]:
    preprocessed = preprocess_query(query)
    ticker = preprocessed.get("ticker") or "TSLA"
    period = preprocessed.get("live_period") or "1mo"
    interval = preprocessed.get("live_interval") or "1d"
    return ticker, period, interval


def _tool_name_is_historical(tool_name: str) -> bool:
    lowered = (tool_name or "").lower()
    return any(token in lowered for token in ["historical", "price_history", "stock_prices"])


def _fallback_live_finance_by_yfinance(query: str, reason: str | None = None) -> str:
    assert_runtime_ready(
        stage="mcp_tools._fallback_live_finance_by_yfinance",
        packages=["yfinance"],
    )
    import yfinance as yf

    ticker, period, interval = _extract_market_params(query)
    hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    if hist is None or hist.empty:
        prefix = (
            f"MCP failed ({reason}). " if reason else ""
        )
        return (
            f"{prefix}Fallback yfinance also returned empty data "
            f"for ticker={ticker}, period={period}, interval={interval}."
        )

    close = hist["Close"].dropna()
    if close.empty:
        prefix = (
            f"MCP failed ({reason}). " if reason else ""
        )
        return f"{prefix}Fallback yfinance has no close-price data for {ticker}."

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    change_pct = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
    high = float(close.max())
    low = float(close.min())
    trend = "uptrend" if change_pct > 1 else "downtrend" if change_pct < -1 else "sideways"

    prefix = f"MCP failed ({reason}).\n\n" if reason else ""
    return (
        f"{prefix}Fallback source: yfinance\n"
        f"- Ticker: {ticker}\n"
        f"- Period/Interval: {period} / {interval}\n"
        f"- Start Close: {start_price:.2f}\n"
        f"- End Close: {end_price:.2f}\n"
        f"- Change: {change_pct:.2f}%\n"
        f"- High/Low: {high:.2f} / {low:.2f}\n"
        f"- Trend: {trend}\n"
    )


def get_finance_researcher_tool():
    assert_runtime_ready(
        stage="mcp_tools.get_finance_researcher_tool",
        packages=["langchain_core"],
    )
    from langchain_core.tools import tool

    return tool(finance_researcher)


def get_live_finance_researcher_tool():
    assert_runtime_ready(
        stage="mcp_tools.get_live_finance_researcher_tool",
        packages=["langchain_core"],
    )
    from langchain_core.tools import tool

    return tool(live_finance_researcher)
