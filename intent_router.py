"""Hybrid intent router with deterministic rules plus semantic enrichment."""

from __future__ import annotations

from query_preprocessor import preprocess_query, semantic_enrich_query, split_hybrid_query

FINANCE_GENERAL_HITS = [
    "finance",
    "financial",
    "stock",
    "stocks",
    "market",
    "equity",
    "investment",
    "invest",
    "portfolio",
    "trading",
    "company",
    "earnings",
    "valuation",
    "公司",
    "股票",
    "股市",
    "投资",
    "市场",
    "证券",
    "估值",
    "盈利",
    "财务",
    "财报",
    "基金",
    "风险",
]

LIVE_HITS = [
    "最新",
    "最近",
    "实时",
    "今天",
    "当前",
    "current",
    "latest",
    "today",
    "stock price",
    "price",
    "股价",
    "走势",
    "trend",
    "quote",
    "period",
    "interval",
]

HISTORICAL_HITS = [
    "财报",
    "年报",
    "季报",
    "10-k",
    "10-q",
    "annual report",
    "quarterly report",
    "revenue",
    "net income",
    "earnings",
    "margin",
    "利润",
    "营收",
    "现金流",
    "cash flow",
    "财务",
]

CLARIFICATION_REPLY = (
    "我先不直接拒答。这个问题可能和金融研究有关，但信息还不够明确。"
    "你可以补充公司名/股票代码、时间范围，或说明你想看财报分析还是实时行情。"
)


def route_query(query: str, preprocessed: dict | None = None) -> dict:
    pre = preprocessed or preprocess_query(query)
    text = (pre.get("normalized_query") or query or "").lower()

    live_score = sum(1 for token in LIVE_HITS if token in text)
    rag_score = sum(1 for token in HISTORICAL_HITS if token in text)
    finance_general_score = sum(1 for token in FINANCE_GENERAL_HITS if token in text)

    if pre.get("ticker") and pre.get("live_period"):
        live_score += 2
    if pre.get("fiscal_year") or pre.get("fiscal_quarter"):
        rag_score += 2
    if pre.get("canonical_company") or pre.get("ticker"):
        finance_general_score += 2
    if pre.get("translated_terms"):
        finance_general_score += 1

    rule_intent = _resolve_rule_intent(
        live_score=live_score,
        rag_score=rag_score,
        finance_general_score=finance_general_score,
    )
    semantic = {}
    if _needs_semantic_review(pre, rule_intent, live_score, rag_score, finance_general_score):
        semantic = semantic_enrich_query(query, pre)

    final_intent, needs_clarification, clarification_question = _merge_intents(
        rule_intent=rule_intent,
        semantic=semantic,
        live_score=live_score,
        rag_score=rag_score,
        finance_general_score=finance_general_score,
    )
    recommended_tools = _tools_for_intent(final_intent)
    if needs_clarification:
        recommended_tools = []

    return {
        "intent": final_intent,
        "recommended_tools": recommended_tools,
        "live_score": live_score,
        "rag_score": rag_score,
        "finance_general_score": finance_general_score,
        "rule_intent": rule_intent,
        "semantic_intent": semantic.get("intent_hint"),
        "semantic_confidence": semantic.get("confidence"),
        "semantic_source": semantic.get("analysis_source"),
        "semantic_finance_related": semantic.get("is_finance_related"),
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }


def build_agent_routing_note(query: str, preprocessed: dict | None = None, route: dict | None = None) -> str:
    pre = preprocessed or preprocess_query(query)
    routing = route or route_query(query, pre)
    hybrid_split = {}
    if routing.get("intent") == "hybrid":
        semantic = {}
        if routing.get("semantic_intent") or routing.get("semantic_confidence"):
            semantic = {
                "intent_hint": routing.get("semantic_intent"),
                "confidence": routing.get("semantic_confidence"),
            }
        hybrid_split = split_hybrid_query(query, preprocessed=pre, semantic=semantic or None)

    fields = [
        f"intent={routing.get('intent')}",
        f"recommended_tools={','.join(routing.get('recommended_tools', []))}",
        f"rule_intent={routing.get('rule_intent')}",
        f"semantic_intent={routing.get('semantic_intent')}",
        f"needs_clarification={routing.get('needs_clarification')}",
    ]
    if pre.get("canonical_company"):
        fields.append(f"company={pre['canonical_company']}")
    if pre.get("ticker"):
        fields.append(f"ticker={pre['ticker']}")
    if pre.get("fiscal_year"):
        fields.append(f"fiscal_year={pre['fiscal_year']}")
    if pre.get("fiscal_quarter"):
        fields.append(f"fiscal_quarter={pre['fiscal_quarter']}")
    if pre.get("live_period"):
        fields.append(f"live_period={pre['live_period']}")
    if pre.get("live_interval"):
        fields.append(f"live_interval={pre['live_interval']}")
    if pre.get("report_query_en"):
        fields.append(f"report_query_en={pre['report_query_en']}")
    if pre.get("market_query_en"):
        fields.append(f"market_query_en={pre['market_query_en']}")
    clarification_question = routing.get("clarification_question")
    if clarification_question:
        fields.append(f"clarification_question={clarification_question}")
    if hybrid_split.get("document_query"):
        fields.append(f"document_query={hybrid_split['document_query']}")
    if hybrid_split.get("market_query"):
        fields.append(f"market_query={hybrid_split['market_query']}")

    instructions = (
        "Runtime routing hints from deterministic preprocessing plus semantic confirmation. "
        "Use them as high-priority guidance for tool selection, but only answer with actual tool evidence."
    )
    if routing.get("needs_clarification"):
        instructions += " Ask a clarification question before giving a substantive answer."
    elif routing.get("intent") == "non_finance":
        instructions += " Politely refuse and redirect the user to finance-related requests."
    elif routing.get("intent") == "hybrid":
        instructions += (
            " For hybrid questions, use document_query for deep_rag_search and market_query "
            "for live_finance_researcher, then synthesize both sources explicitly."
        )

    return instructions + "\n" + "\n".join(f"- {item}" for item in fields)


def _resolve_rule_intent(*, live_score: int, rag_score: int, finance_general_score: int) -> str:
    if live_score == 0 and rag_score == 0 and finance_general_score == 0:
        return "non_finance"
    if live_score > 0 and rag_score == 0 and finance_general_score == 0:
        return "non_finance"
    if live_score > 0 and rag_score > 0:
        return "hybrid"
    if live_score > 0:
        return "live_market"
    if rag_score > 0:
        return "historical_rag"
    return "finance_general"


def _needs_semantic_review(
    preprocessed: dict,
    rule_intent: str,
    live_score: int,
    rag_score: int,
    finance_general_score: int,
) -> bool:
    if rule_intent == "non_finance":
        return True
    if preprocessed.get("is_mixed_language"):
        return True
    if live_score > 0 and rag_score > 0:
        return True
    if finance_general_score <= 1:
        return True
    return False


def _merge_intents(
    *,
    rule_intent: str,
    semantic: dict,
    live_score: int,
    rag_score: int,
    finance_general_score: int,
) -> tuple[str, bool, str]:
    clarification_question = semantic.get("clarification_question") or CLARIFICATION_REPLY
    semantic_intent = str(semantic.get("intent_hint") or "").strip().lower()
    semantic_confidence = str(semantic.get("confidence") or "").strip().lower()
    is_finance_related = semantic.get("is_finance_related")
    needs_clarification = bool(semantic.get("needs_clarification"))

    finance_intents = {"historical_rag", "live_market", "hybrid", "finance_general"}

    if rule_intent == "non_finance":
        if semantic_intent in finance_intents or is_finance_related is True:
            if semantic_intent in finance_intents:
                return semantic_intent, needs_clarification, clarification_question
            return "finance_general", True, clarification_question
        if needs_clarification or semantic_confidence != "high":
            return "finance_general", True, clarification_question
        return "non_finance", False, ""

    if semantic_intent == "hybrid":
        return "hybrid", needs_clarification, clarification_question
    if semantic_intent in finance_intents and semantic_confidence == "high":
        return semantic_intent, needs_clarification, clarification_question

    if (
        semantic_intent == "non_finance"
        and is_finance_related is False
        and semantic_confidence == "high"
        and live_score == 0
        and rag_score == 0
        and finance_general_score == 0
    ):
        return "non_finance", False, ""

    return rule_intent, needs_clarification, clarification_question if needs_clarification else ""


def _tools_for_intent(intent: str) -> list[str]:
    mapping = {
        "historical_rag": ["deep_rag_search"],
        "live_market": ["live_finance_researcher"],
        "hybrid": ["deep_rag_search", "live_finance_researcher"],
        "finance_general": ["deep_rag_search"],
        "non_finance": [],
    }
    return mapping.get(intent, ["deep_rag_search"])
