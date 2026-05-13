"""Query preprocessing split into deterministic extraction and semantic enrichment."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import io
import json
import re
import warnings

COMPANY_ALIAS_TO_CANONICAL = {
    "apple": "apple",
    "苹果": "apple",
    "aapl": "apple",
    "tesla": "tesla",
    "特斯拉": "tesla",
    "tsla": "tesla",
    "microsoft": "microsoft",
    "微软": "microsoft",
    "msft": "microsoft",
    "amazon": "amazon",
    "亚马逊": "amazon",
    "amzn": "amazon",
    "nvidia": "nvidia",
    "英伟达": "nvidia",
    "英伟达公司": "nvidia",
    "nvda": "nvidia",
    "google": "alphabet",
    "alphabet": "alphabet",
    "谷歌": "alphabet",
    "googl": "alphabet",
    "goog": "alphabet",
    "meta": "meta",
    "facebook": "meta",
    "脸书": "meta",
    "meta platforms": "meta",
    "meta platform": "meta",
    "腾讯": "tencent",
    "tencent": "tencent",
    "阿里": "alibaba",
    "阿里巴巴": "alibaba",
    "alibaba": "alibaba",
    "baba": "alibaba",
}

CANONICAL_TO_TICKER = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "meta": "META",
    "tencent": "0700.HK",
    "alibaba": "BABA",
}

TERM_TRANSLATIONS = {
    "营收": "revenue",
    "收入": "revenue",
    "销售额": "sales",
    "净利润": "net income",
    "利润": "profit",
    "盈利": "earnings",
    "业绩": "financial performance",
    "每股收益": "earnings per share",
    "eps": "earnings per share",
    "市盈率": "pe ratio",
    "估值": "valuation",
    "市值": "market capitalization",
    "毛利率": "gross margin",
    "营业利润率": "operating margin",
    "利润率": "margin",
    "现金流": "cash flow",
    "自由现金流": "free cash flow",
    "资产负债表": "balance sheet",
    "利润表": "income statement",
    "损益表": "income statement",
    "现金流量表": "cash flow statement",
    "指引": "guidance",
    "展望": "outlook",
    "风险": "risk",
    "竞争": "competition",
    "财报": "financial report",
    "年报": "annual report",
    "季报": "quarterly report",
    "电话会": "earnings call",
    "股价": "stock price",
    "走势": "price trend",
    "涨跌": "price change",
    "回撤": "drawdown",
}

HISTORICAL_INTENT_TERMS = [
    "10-k",
    "10-q",
    "annual report",
    "quarterly report",
    "financial statements",
    "财报",
    "年报",
    "季报",
    "电话会",
    "利润",
    "营收",
    "现金流",
    "毛利率",
    "风险",
    "指引",
]

LIVE_INTENT_TERMS = [
    "最新",
    "最近",
    "实时",
    "今天",
    "当前",
    "盘中",
    "现价",
    "quote",
    "current",
    "latest",
    "today",
    "intraday",
    "price",
    "trend",
    "走势",
    "股价",
]

FINANCE_SIGNAL_TERMS = [
    "finance",
    "financial",
    "stock",
    "stocks",
    "market",
    "trading",
    "investment",
    "equity",
    "earnings",
    "revenue",
    "profit",
    "margin",
    "cash flow",
    "valuation",
    "market cap",
    "10-k",
    "10-q",
    "财报",
    "年报",
    "季报",
    "股票",
    "股价",
    "股市",
    "财务",
    "营收",
    "利润",
    "估值",
    "市值",
    "现金流",
    "投资",
    "风险",
]

HYBRID_DOCUMENT_AVOID_TERMS = [
    "stock price",
    "price trend",
    "quote",
    "intraday",
    "today",
    "latest",
]

HYBRID_MARKET_AVOID_TERMS = [
    "10-k",
    "10-q",
    "annual report",
    "quarterly report",
    "financial statements",
    "risk factors",
]

HYBRID_DOCUMENT_METRIC_TERMS = [
    "revenue",
    "profit",
    "net income",
    "margin",
    "cash flow",
    "guidance",
    "risk",
]


def preprocess_query(query: str) -> dict:
    """Deterministically extract structured fields from the raw user query."""
    raw_query = query or ""
    normalized_query = _normalize_whitespace(raw_query)
    lowered = normalized_query.lower()

    canonical_company = _extract_canonical_company(normalized_query, lowered)
    ticker = _extract_ticker(normalized_query, lowered, canonical_company)
    fiscal_year = _extract_fiscal_year(normalized_query)
    fiscal_quarter = _extract_fiscal_quarter(lowered)
    live_period, live_interval = _extract_live_time_params(normalized_query, lowered)
    translated_terms = _extract_translated_terms(normalized_query)
    language = _detect_language(normalized_query)
    is_mixed_language = language == "mixed"
    english_company = canonical_company
    english_core_query = _build_english_core_query(
        canonical_company=canonical_company,
        ticker=ticker,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        translated_terms=translated_terms,
        normalized_query=normalized_query,
    )
    market_query_en = _build_market_query_en(
        canonical_company=canonical_company,
        ticker=ticker,
        live_period=live_period,
        live_interval=live_interval,
        translated_terms=translated_terms,
        normalized_query=normalized_query,
    )
    report_query_en = _build_report_query_en(
        canonical_company=canonical_company,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        translated_terms=translated_terms,
        normalized_query=normalized_query,
    )

    bilingual_query = normalized_query
    if translated_terms:
        bilingual_query = f"{normalized_query} | {' | '.join(translated_terms)}"

    return {
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "canonical_company": canonical_company,
        "english_company": english_company,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "live_period": live_period,
        "live_interval": live_interval,
        "translated_terms": translated_terms,
        "bilingual_query": bilingual_query,
        "detected_language": language,
        "is_mixed_language": is_mixed_language,
        "english_core_query": english_core_query,
        "market_query_en": market_query_en,
        "report_query_en": report_query_en,
    }


def build_retrieval_queries(preprocessed: dict) -> list[str]:
    queries = [
        preprocessed.get("raw_query", ""),
        preprocessed.get("normalized_query", ""),
        preprocessed.get("bilingual_query", ""),
        preprocessed.get("english_core_query", ""),
        preprocessed.get("report_query_en", ""),
    ]

    canonical_company = preprocessed.get("canonical_company")
    fiscal_year = preprocessed.get("fiscal_year")
    fiscal_quarter = preprocessed.get("fiscal_quarter")
    translated_terms = preprocessed.get("translated_terms") or []

    if canonical_company:
        english_parts = [canonical_company]
        if fiscal_year:
            english_parts.append(str(fiscal_year))
        if fiscal_quarter:
            english_parts.append(fiscal_quarter.upper())
        english_parts.extend(translated_terms)
        if english_parts:
            queries.append(" ".join(english_parts))

    return dedupe_queries(queries)


def build_live_queries(preprocessed: dict) -> list[str]:
    queries = [
        preprocessed.get("raw_query", ""),
        preprocessed.get("normalized_query", ""),
        preprocessed.get("bilingual_query", ""),
        preprocessed.get("market_query_en", ""),
    ]
    ticker = preprocessed.get("ticker")
    live_period = preprocessed.get("live_period")
    live_interval = preprocessed.get("live_interval")
    if ticker:
        parts = [ticker, "stock price"]
        if live_period:
            parts.append(live_period)
        if live_interval:
            parts.append(live_interval)
        queries.append(" ".join(parts))
    return dedupe_queries(queries)


def semantic_enrich_query(query: str, preprocessed: dict | None = None) -> dict:
    """Use an LLM when available to enrich intent and English rewrites."""
    pre = preprocessed or preprocess_query(query)
    normalized = pre.get("normalized_query") or _normalize_whitespace(query or "")
    if not normalized:
        return _fallback_semantic_expansion(pre)
    return _semantic_query_expansion_cached(normalized)


def get_semantic_query_expansion(query: str, preprocessed: dict | None = None) -> dict:
    """Backward-compatible alias for semantic enrichment."""
    return semantic_enrich_query(query, preprocessed)


def split_hybrid_query(
    query: str,
    preprocessed: dict | None = None,
    semantic: dict | None = None,
) -> dict:
    """Split a hybrid query into document-oriented and market-oriented subqueries."""
    pre = preprocessed or preprocess_query(query)
    sem = semantic or semantic_enrich_query(query, pre)
    normalized = pre.get("normalized_query") or _normalize_whitespace(query or "")
    if not normalized:
        return _fallback_hybrid_split(pre, sem)
    return _hybrid_query_split_cached(normalized)


def dedupe_queries(items: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        cleaned = _normalize_whitespace(str(item))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


@lru_cache(maxsize=256)
def _semantic_query_expansion_cached(query: str) -> dict:
    pre = preprocess_query(query)
    prompt = f"""
You normalize bilingual finance queries for two downstream systems:
1. English SEC-style financial-report retrieval
2. English Yahoo Finance market-data tools

Return JSON only with this schema:
{{
  "is_finance_related": true,
  "intent_hint": "historical_rag|live_market|hybrid|finance_general|non_finance|unknown",
  "confidence": "high|medium|low",
  "english_query": "single concise English paraphrase",
  "search_queries": ["query 1", "query 2"],
  "finance_terms": ["term 1", "term 2"],
  "needs_clarification": false,
  "clarification_question": ""
}}

Rules:
- Preserve the user's meaning.
- If the query is Chinese or mixed Chinese-English, rewrite it into natural finance English.
- Prefer terminology likely to appear in 10-K / 10-Q filings when the request is document-oriented.
- Prefer stock-market wording when the request is live-market oriented.
- If the question is clearly outside finance/investing/public-company research, set is_finance_related=false.
- If the query is ambiguous but still possibly finance-related, set needs_clarification=true instead of non_finance.

User query: {query}
Known normalization:
- company={pre.get("canonical_company")}
- ticker={pre.get("ticker")}
- fiscal_year={pre.get("fiscal_year")}
- fiscal_quarter={pre.get("fiscal_quarter")}
- translated_terms={pre.get("translated_terms")}
- report_query_en={pre.get("report_query_en")}
- market_query_en={pre.get("market_query_en")}
""".strip()
    try:
        response = _quiet_invoke_llm(prompt)
        content = getattr(response, "content", response)
        data = _load_json_from_response(content)
        if not isinstance(data, dict):
            raise ValueError("semantic expansion response is not a JSON object")

        search_queries = data.get("search_queries", [])
        if not isinstance(search_queries, list):
            search_queries = []
        finance_terms = data.get("finance_terms", [])
        if not isinstance(finance_terms, list):
            finance_terms = []

        english_query = _normalize_whitespace(data.get("english_query") or "")
        fallback = _fallback_semantic_expansion(pre)
        return {
            "is_finance_related": bool(data.get("is_finance_related", True)),
            "intent_hint": str(data.get("intent_hint") or fallback["intent_hint"]).strip().lower(),
            "confidence": str(data.get("confidence") or "low").strip().lower(),
            "english_query": english_query or fallback["english_query"],
            "search_queries": dedupe_queries([*search_queries, *fallback["search_queries"]])[:4],
            "finance_terms": dedupe_queries([*finance_terms, *fallback["finance_terms"]])[:8],
            "needs_clarification": bool(data.get("needs_clarification", False)),
            "clarification_question": _normalize_whitespace(
                str(data.get("clarification_question") or "")
            ),
            "analysis_source": "llm",
        }
    except Exception:  # noqa: BLE001
        fallback = _fallback_semantic_expansion(pre)
        fallback["analysis_source"] = "fallback"
        return fallback


@lru_cache(maxsize=256)
def _hybrid_query_split_cached(query: str) -> dict:
    pre = preprocess_query(query)
    sem = semantic_enrich_query(query, pre)
    prompt = f"""
You are splitting a finance question into two subqueries for a hybrid workflow:
1. A document-oriented query for English 10-K / 10-Q / annual report retrieval
2. A market-oriented query for Yahoo Finance style stock tools

Return JSON only:
{{
  "document_query": "short English document query",
  "market_query": "short English market query",
  "document_focus": ["term1", "term2"],
  "market_focus": ["term1", "term2"]
}}

Rules:
- Preserve the user's intent.
- The document query should emphasize historical filings, metrics, guidance, risks, margins, or business performance.
- The market query should emphasize ticker, price trend, quote, period, interval, market cap, or valuation.
- If the user asks a comparison between financial performance and recent stock performance, split them cleanly.
- Use concise natural English.
- Do not put stock-price-only wording into document_query unless the user explicitly asks for hybrid comparison.
- Do not put filing-specific wording like 10-K, 10-Q, annual report, or quarterly report into market_query.

User query: {query}
Known normalization:
- company={pre.get("canonical_company")}
- ticker={pre.get("ticker")}
- fiscal_year={pre.get("fiscal_year")}
- fiscal_quarter={pre.get("fiscal_quarter")}
- live_period={pre.get("live_period")}
- live_interval={pre.get("live_interval")}
- english_query={sem.get("english_query")}
- report_query_en={pre.get("report_query_en")}
- market_query_en={pre.get("market_query_en")}
""".strip()
    try:
        response = _quiet_invoke_llm(prompt)
        content = getattr(response, "content", response)
        data = _load_json_from_response(content)
        if not isinstance(data, dict):
            raise ValueError("hybrid split response is not a JSON object")
        fallback = _fallback_hybrid_split(pre, sem)
        document_query = _normalize_whitespace(data.get("document_query") or "")
        market_query = _normalize_whitespace(data.get("market_query") or "")
        document_focus = data.get("document_focus", [])
        market_focus = data.get("market_focus", [])
        if not isinstance(document_focus, list):
            document_focus = []
        if not isinstance(market_focus, list):
            market_focus = []
        sanitized = _sanitize_hybrid_split(
            preprocessed=pre,
            semantic=sem,
            document_query=document_query or fallback["document_query"],
            market_query=market_query or fallback["market_query"],
            document_focus=document_focus,
            market_focus=market_focus,
        )
        return {
            "document_query": sanitized["document_query"],
            "market_query": sanitized["market_query"],
            "document_candidates": dedupe_queries(
                [
                    sanitized["document_query"],
                    fallback["document_query"],
                    *fallback["document_candidates"],
                ]
            )[:4],
            "market_candidates": dedupe_queries(
                [
                    sanitized["market_query"],
                    fallback["market_query"],
                    *fallback["market_candidates"],
                ]
            )[:4],
            "document_focus": sanitized["document_focus"],
            "market_focus": sanitized["market_focus"],
            "analysis_source": "llm",
        }
    except Exception:  # noqa: BLE001
        fallback = _fallback_hybrid_split(pre, sem)
        fallback["analysis_source"] = "fallback"
        return fallback


def _fallback_semantic_expansion(preprocessed: dict) -> dict:
    normalized_query = preprocessed.get("normalized_query", "")
    english_query = (
        preprocessed.get("report_query_en")
        or preprocessed.get("market_query_en")
        or preprocessed.get("english_core_query")
        or normalized_query
    )
    intent_hint = _infer_intent_from_preprocessed(preprocessed)
    return {
        "is_finance_related": _has_finance_signal(preprocessed),
        "intent_hint": intent_hint,
        "confidence": "medium" if english_query else "low",
        "english_query": english_query,
        "search_queries": dedupe_queries(
            [
                preprocessed.get("english_core_query", ""),
                preprocessed.get("report_query_en", ""),
                preprocessed.get("market_query_en", ""),
                normalized_query,
            ]
        )[:4],
        "finance_terms": preprocessed.get("translated_terms") or [],
        "needs_clarification": False,
        "clarification_question": "",
    }


def _fallback_hybrid_split(preprocessed: dict, semantic: dict) -> dict:
    normalized = preprocessed.get("normalized_query", "")
    document_candidates = dedupe_queries(
        [
            preprocessed.get("report_query_en", ""),
            preprocessed.get("english_core_query", ""),
            semantic.get("english_query", ""),
            normalized,
        ]
    )
    market_candidates = dedupe_queries(
        [
            preprocessed.get("market_query_en", ""),
            preprocessed.get("ticker", ""),
            normalized,
        ]
    )
    document_focus = dedupe_queries(
        [
            *(preprocessed.get("translated_terms") or []),
            *(semantic.get("finance_terms") or []),
            "financial report" if preprocessed.get("report_query_en") else "",
        ]
    )[:6]
    market_focus = dedupe_queries(
        [
            preprocessed.get("ticker", ""),
            "stock price" if preprocessed.get("market_query_en") else "",
            preprocessed.get("live_period", ""),
            preprocessed.get("live_interval", ""),
        ]
    )[:6]
    sanitized = _sanitize_hybrid_split(
        preprocessed=preprocessed,
        semantic=semantic,
        document_query=document_candidates[0] if document_candidates else normalized,
        market_query=market_candidates[0] if market_candidates else normalized,
        document_focus=document_focus,
        market_focus=market_focus,
    )
    return {
        "document_query": sanitized["document_query"],
        "market_query": sanitized["market_query"],
        "document_candidates": document_candidates[:4],
        "market_candidates": market_candidates[:4],
        "document_focus": sanitized["document_focus"],
        "market_focus": sanitized["market_focus"],
    }


def _sanitize_hybrid_split(
    *,
    preprocessed: dict,
    semantic: dict,
    document_query: str,
    market_query: str,
    document_focus: list[str],
    market_focus: list[str],
) -> dict:
    document_query = _normalize_whitespace(document_query)
    market_query = _normalize_whitespace(market_query)

    for token in HYBRID_DOCUMENT_AVOID_TERMS:
        if token in document_query.lower():
            document_query = _remove_term(document_query, token)
    for token in HYBRID_MARKET_AVOID_TERMS:
        if token in market_query.lower():
            market_query = _remove_term(market_query, token)

    document_query = _normalize_whitespace(document_query)
    market_query = _normalize_whitespace(market_query)

    if not document_query:
        document_query = (
            preprocessed.get("report_query_en")
            or preprocessed.get("english_core_query")
            or semantic.get("english_query")
            or preprocessed.get("normalized_query")
            or ""
        )
    if not market_query:
        market_query = (
            preprocessed.get("market_query_en")
            or preprocessed.get("ticker")
            or preprocessed.get("normalized_query")
            or ""
        )

    if preprocessed.get("canonical_company") and preprocessed["canonical_company"] not in document_query.lower():
        document_query = _normalize_whitespace(
            f"{preprocessed['canonical_company']} {document_query}".strip()
        )
    if preprocessed.get("ticker") and preprocessed["ticker"].lower() not in market_query.lower():
        market_query = _normalize_whitespace(f"{preprocessed['ticker']} {market_query}".strip())
    if preprocessed.get("fiscal_year"):
        year_text = str(preprocessed["fiscal_year"])
        if year_text not in document_query:
            document_query = _normalize_whitespace(f"{document_query} {year_text}")

    normalized_query = (preprocessed.get("normalized_query") or "").lower()
    if _hybrid_document_query_too_generic(document_query):
        anchor_terms = preprocessed.get("translated_terms") or []
        anchor_terms = [term for term in anchor_terms if term in HYBRID_DOCUMENT_METRIC_TERMS]
        if not anchor_terms:
            anchor_terms = ["revenue", "profit", "margin", "cash flow"]
        document_query = _normalize_whitespace(f"{document_query} {' '.join(anchor_terms[:4])}")

    if _needs_recent_market_window(normalized_query, preprocessed, market_query):
        market_query = _normalize_whitespace(f"{market_query} 1mo")

    sanitized_document_focus = [
        item
        for item in dedupe_queries(document_focus)
        if item.lower() not in HYBRID_DOCUMENT_AVOID_TERMS
    ][:6]
    sanitized_market_focus = [
        item
        for item in dedupe_queries(market_focus)
        if item.lower() not in HYBRID_MARKET_AVOID_TERMS
    ][:6]

    return {
        "document_query": document_query,
        "market_query": market_query,
        "document_focus": sanitized_document_focus,
        "market_focus": sanitized_market_focus,
    }


def _hybrid_document_query_too_generic(document_query: str) -> bool:
    lowered = document_query.lower()
    if not lowered:
        return True
    has_metric_term = any(term in lowered for term in HYBRID_DOCUMENT_METRIC_TERMS)
    is_generic_performance = any(
        token in lowered
        for token in ["financial performance", "business performance", "company performance"]
    )
    return is_generic_performance and not has_metric_term


def _needs_recent_market_window(normalized_query: str, preprocessed: dict, market_query: str) -> bool:
    if preprocessed.get("live_period"):
        return False
    lowered_market = market_query.lower()
    if any(token in lowered_market for token in ["1mo", "3mo", "6mo", "1y", "5y", "1wk"]):
        return False
    recent_signals = ["最近", "近期", "recent", "last", "past"]
    market_signals = ["股价", "price", "trend", "走势", "stock price"]
    return any(token in normalized_query for token in recent_signals) and any(
        token in normalized_query for token in market_signals
    )


def _quiet_invoke_llm(prompt: str):
    from model_factory import get_chat_model

    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
        with redirect_stdout(sink), redirect_stderr(sink):
            return get_chat_model(temperature=0).invoke(prompt)


def _infer_intent_from_preprocessed(preprocessed: dict) -> str:
    if not _has_finance_signal(preprocessed):
        return "unknown"
    normalized = (preprocessed.get("normalized_query") or "").lower()
    live_score = sum(1 for token in LIVE_INTENT_TERMS if token in normalized)
    historical_score = sum(1 for token in HISTORICAL_INTENT_TERMS if token in normalized)
    if preprocessed.get("live_period"):
        live_score += 2
    if preprocessed.get("fiscal_year") or preprocessed.get("fiscal_quarter"):
        historical_score += 2
    if live_score > 0 and historical_score > 0:
        return "hybrid"
    if live_score > 0:
        return "live_market"
    if historical_score > 0:
        return "historical_rag"
    if preprocessed.get("canonical_company") or preprocessed.get("ticker"):
        return "finance_general"
    return "unknown"


def _has_finance_signal(preprocessed: dict) -> bool:
    if preprocessed.get("canonical_company") or preprocessed.get("ticker"):
        return True
    if preprocessed.get("translated_terms"):
        return True
    normalized = (preprocessed.get("normalized_query") or "").lower()
    return any(token in normalized for token in FINANCE_SIGNAL_TERMS)


def _build_english_core_query(
    *,
    canonical_company: str | None,
    ticker: str | None,
    fiscal_year: int | None,
    fiscal_quarter: str | None,
    translated_terms: list[str],
    normalized_query: str,
) -> str:
    parts = []
    if canonical_company:
        parts.append(canonical_company)
    if ticker and ticker not in parts:
        parts.append(ticker)
    if fiscal_year:
        parts.append(str(fiscal_year))
    if fiscal_quarter:
        parts.append(fiscal_quarter.upper())
    parts.extend(translated_terms[:4])
    if not parts:
        return normalized_query
    return " ".join(parts)


def _build_market_query_en(
    *,
    canonical_company: str | None,
    ticker: str | None,
    live_period: str | None,
    live_interval: str | None,
    translated_terms: list[str],
    normalized_query: str,
) -> str:
    base = canonical_company or ticker
    if not base:
        return ""
    parts = [base, "stock price"]
    if "price trend" in translated_terms or "stock price" in translated_terms:
        parts.append("trend")
    if "market capitalization" in translated_terms:
        parts.append("market cap")
    if live_period:
        parts.append(live_period)
    if live_interval:
        parts.append(live_interval)
    if any(token in normalized_query.lower() for token in ["最新", "当前", "current", "latest"]):
        parts.append("latest")
    return " ".join(parts)


def _build_report_query_en(
    *,
    canonical_company: str | None,
    fiscal_year: int | None,
    fiscal_quarter: str | None,
    translated_terms: list[str],
    normalized_query: str,
) -> str:
    if not canonical_company:
        return ""
    parts = [canonical_company]
    if fiscal_year:
        parts.append(str(fiscal_year))
    if fiscal_quarter:
        parts.append(fiscal_quarter.upper())
        parts.extend(["10-Q", "quarterly report"])
    else:
        parts.extend(["10-K", "annual report"])
    parts.extend(translated_terms[:4])
    if not translated_terms and any(token in normalized_query.lower() for token in ["财报", "年报", "季报"]):
        parts.append("financial statements")
    return " ".join(parts)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_term(text: str, term: str) -> str:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    cleaned = pattern.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,|-")


def _detect_language(text: str) -> str:
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_zh and has_en:
        return "mixed"
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "unknown"


def _extract_canonical_company(original: str, lowered: str) -> str | None:
    text_candidates = [original, lowered]
    for candidate in text_candidates:
        for alias, canonical in COMPANY_ALIAS_TO_CANONICAL.items():
            if alias in candidate:
                return canonical
    return None


def _extract_ticker(original: str, lowered: str, canonical_company: str | None) -> str | None:
    explicit = re.search(r"\b[A-Z]{1,8}(?:\.[A-Z]{1,3})?\b", original.upper())
    if explicit:
        token = explicit.group(0)
        if token in CANONICAL_TO_TICKER.values():
            return token

    for alias, canonical in COMPANY_ALIAS_TO_CANONICAL.items():
        if alias in lowered or alias in original:
            return CANONICAL_TO_TICKER.get(canonical)

    if canonical_company:
        return CANONICAL_TO_TICKER.get(canonical_company)
    return None


def _extract_fiscal_year(text: str) -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
    if match:
        return int(match.group(1))
    return None


def _extract_fiscal_quarter(lowered: str) -> str | None:
    for quarter in ("q1", "q2", "q3", "q4"):
        if quarter in lowered:
            return quarter
    zh_map = {
        "一季度": "q1",
        "二季度": "q2",
        "三季度": "q3",
        "四季度": "q4",
        "第一季度": "q1",
        "第二季度": "q2",
        "第三季度": "q3",
        "第四季度": "q4",
    }
    for token, quarter in zh_map.items():
        if token in lowered:
            return quarter
    return None


def _extract_live_time_params(original: str, lowered: str) -> tuple[str | None, str | None]:
    period = None
    interval = None

    if any(key in original for key in ["今天", "今日", "当天"]) or re.search(r"\b(today|current day)\b", lowered) or "1d" in lowered:
        period = "1d"
    elif _matches_live_period(original, lowered, zh_number="1", zh_words=["一个"], en_numbers=["1", "one"], unit="week") or "1wk" in lowered:
        period = "1wk"
    elif _matches_live_period(original, lowered, zh_number="1", zh_words=["一个"], en_numbers=["1", "one"], unit="month") or "1mo" in lowered:
        period = "1mo"
    elif _matches_live_period(original, lowered, zh_number="3", zh_words=["三"], en_numbers=["3", "three"], unit="month") or "3mo" in lowered:
        period = "3mo"
    elif _matches_live_period(original, lowered, zh_number="6", zh_words=["六"], en_numbers=["6", "six"], unit="month") or "6mo" in lowered:
        period = "6mo"
    elif _matches_live_period(original, lowered, zh_number="1", zh_words=["一"], en_numbers=["1", "one"], unit="year") or "1y" in lowered:
        period = "1y"
    elif _matches_live_period(original, lowered, zh_number="5", zh_words=["五"], en_numbers=["5", "five"], unit="year") or "5y" in lowered:
        period = "5y"

    if "分钟" in original or re.search(r"(?<![a-z0-9])1m(?![a-z])", lowered):
        interval = "1m"
    elif "小时" in original or re.search(r"(?<![a-z0-9])1h(?![a-z])", lowered):
        interval = "1h"
    elif "日线" in original or "daily" in lowered or re.search(r"(?<![a-z0-9])1d(?![a-z])", lowered):
        interval = "1d"
    elif "周线" in original or "weekly" in lowered:
        interval = "1wk"
    elif "月线" in original or "monthly" in lowered:
        interval = "1mo"

    return period, interval


def _matches_live_period(
    original: str,
    lowered: str,
    *,
    zh_number: str,
    zh_words: list[str],
    en_numbers: list[str],
    unit: str,
) -> bool:
    zh_units = {
        "week": ["周", "星期"],
        "month": ["个月", "月"],
        "year": ["年"],
    }
    en_units = {
        "week": ["week", "weeks"],
        "month": ["month", "months"],
        "year": ["year", "years"],
    }

    zh_patterns = []
    for number in [zh_number, *zh_words]:
        for unit_text in zh_units[unit]:
            zh_patterns.extend(
                [
                    rf"{number}\s*{unit_text}",
                    rf"(最近|过去|近)\s*{number}\s*{unit_text}",
                ]
            )

    en_patterns = []
    for number in en_numbers:
        for unit_text in en_units[unit]:
            en_patterns.extend(
                [
                    rf"\b{number}\s+{unit_text}\b",
                    rf"\b(last|past|recent)\s+{number}\s+{unit_text}\b",
                    rf"\bfor\s+the\s+last\s+{number}\s+{unit_text}\b",
                ]
            )

    return any(re.search(pattern, original, re.IGNORECASE) for pattern in zh_patterns) or any(
        re.search(pattern, lowered, re.IGNORECASE) for pattern in en_patterns
    )


def _extract_translated_terms(text: str) -> list[str]:
    terms = []
    lowered = text.lower()
    for zh, en in TERM_TRANSLATIONS.items():
        if zh in text and en not in terms:
            terms.append(en)
    for english in [
        "revenue",
        "net income",
        "cash flow",
        "earnings",
        "guidance",
        "outlook",
        "risk",
        "margin",
        "stock price",
        "trend",
        "market cap",
    ]:
        if english in lowered and english not in terms:
            terms.append(english)
    return terms


def _load_json_from_response(content) -> dict:
    text = content
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    text = str(text).strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    return json.loads(text)
