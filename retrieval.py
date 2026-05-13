"""Hybrid retrieval and reranking pipeline."""

from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import io
import json
import logging
import os
import warnings

from model_factory import get_chat_model
from query_preprocessor import (
    build_retrieval_queries,
    dedupe_queries,
    preprocess_query,
    semantic_enrich_query,
)
from runtime_checks import assert_runtime_ready
from vector_store import build_vector_store

FINANCIAL_STRUCTURE_POSITIVE_TERMS = [
    "consolidated statements of operations",
    "consolidated statements of comprehensive income",
    "consolidated balance sheets",
    "consolidated statements of cash flows",
    "net sales",
    "net income",
    "cash flows",
    "gross margin",
    "operating income",
    "dollars in millions",
    "years ended",
]

FINANCIAL_STRUCTURE_NEGATIVE_TERMS = [
    "table of contents",
    "report of independent registered public accounting firm",
    "description of the matter",
    "certifications of chief executive officer",
    "chief financial officer",
    "section 906",
    "sarbanes-oxley",
    "internal control over financial reporting",
    "effectiveness of internal control over financial reporting",
    "i have reviewed this annual report",
    "fairly presents in all material respects",
    "incorporated by reference",
    "exhibit index",
    "form s-8",
    "investor relations website",
]

RETRIEVAL_DOCUMENT_SIGNALS = [
    "10-k",
    "10-q",
    "annual report",
    "quarterly report",
    "financial report",
    "financial statements",
    "statement",
]

RETRIEVAL_SPECIALTY_TERMS = {
    "risk": ["risk factors", "material risks", "cybersecurity risks"],
    "cash flow": [
        "cash flow statement",
        "statements of cash flows",
        "operating cash flow",
        "cash flows",
    ],
    "margin": ["gross margin", "operating margin", "gross profit margin"],
    "revenue": ["revenue", "net sales", "sales by category", "disaggregated revenue"],
    "profit": ["profit", "net income", "operating income", "comprehensive income"],
}

NOISE_PAGE_ROLES = {
    "table_of_contents",
    "financial_index",
    "audit_report",
    "internal_control",
    "certification",
    "website_disclosure",
    "exhibit_index",
}

STRUCTURED_PAGE_ROLES = {
    "risk_factors",
    "cash_flow_statement",
    "income_statement",
    "balance_sheet",
    "shareholders_equity",
    "gross_margin_discussion",
    "md_and_a",
}

RERANK_ROLE_PREFERENCES = {
    "risk": {"risk_factors": 2.0, "md_and_a": 0.8, "gross_margin_discussion": 0.2},
    "cash_flow": {"cash_flow_statement": 2.2, "md_and_a": 0.7, "balance_sheet": 0.2},
    "margin": {"gross_margin_discussion": 1.6, "income_statement": 1.1, "md_and_a": 0.7},
    "revenue": {"income_statement": 1.3, "gross_margin_discussion": 0.9, "md_and_a": 0.7},
    "profit": {"income_statement": 1.4, "md_and_a": 0.8, "gross_margin_discussion": 0.6},
}

RERANK_ROLE_PENALTIES = {
    "risk": {"financial_index": -1.2, "income_statement": -0.9, "cash_flow_statement": -0.4},
    "cash_flow": {"financial_index": -1.8, "income_statement": -1.0, "gross_margin_discussion": -0.6},
    "margin": {"financial_index": -1.0, "cash_flow_statement": -0.7},
    "revenue": {"financial_index": -0.9, "cash_flow_statement": -0.6},
    "profit": {"financial_index": -0.9, "cash_flow_statement": -0.5},
}

RERANK_CONTENT_PREFERENCES = {
    "risk": [
        "risk factors",
        "materially and adversely affect",
        "supply chain",
        "cybersecurity",
        "macroeconomic",
        "foreign exchange",
    ],
    "cash_flow": [
        "operating activities",
        "investing activities",
        "financing activities",
        "net cash provided by",
        "cash, cash equivalents and restricted cash",
        "statements of cash flows",
    ],
    "margin": [
        "gross margin",
        "gross margin percentage",
        "products and services gross margin",
        "operating margin",
    ],
    "revenue": [
        "net sales",
        "revenue",
        "sales by category",
        "disaggregated by significant products and services",
    ],
    "profit": [
        "net income",
        "operating income",
        "comprehensive income",
        "profit",
    ],
}

RERANK_CONTENT_PENALTIES = {
    "risk": ["net sales", "gross margin percentage", "disaggregated by significant products and services"],
    "cash_flow": ["index to consolidated financial statements", "net sales", "gross margin"],
}


def _build_chunk_metadata_model():
    from typing import Optional

    from pydantic import BaseModel, Field

    class ChunkMetadata(BaseModel):
        company_name: Optional[str] = Field(None, description="Company name (lowercase)")
        doc_type: Optional[str] = Field(None, description="Document type: 10-k, 10-q")
        fiscal_year: Optional[int] = Field(None, description="Fiscal year")
        fiscal_quarter: Optional[str] = Field(None, description="Fiscal quarter: q1-q4")

    return ChunkMetadata


@lru_cache(maxsize=1)
def get_llm():
    assert_runtime_ready(
        stage="retrieval.get_llm",
        packages=["pydantic"],
    )
    return get_chat_model()


@lru_cache(maxsize=1)
def get_vector_store():
    return build_vector_store()


@lru_cache(maxsize=1)
def get_reranker():
    assert_runtime_ready(
        stage="retrieval.get_reranker",
        packages=["langchain_community", "sentence_transformers", "torch"],
    )
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for logger_name in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")


def extract_filters(query: str) -> dict:
    preprocessed = preprocess_query(query)
    semantic = semantic_enrich_query(query, preprocessed)
    chunk_metadata_model = _build_chunk_metadata_model()
    structured_llm = get_llm().with_structured_output(chunk_metadata_model)
    prompt = f"""
You are an expert at extracting structured filters from financial queries.
Return JSON fields: company_name, doc_type, fiscal_year, fiscal_quarter.
Return null for missing fields.
Normalize company names to lowercase.
Normalize doc types to 10-k or 10-q.
Normalize quarter to q1, q2, q3, q4.
Query: {semantic.get('english_query') or preprocessed.get('bilingual_query') or query}
""".strip()
    try:
        metadata = _quiet_call(structured_llm.invoke, prompt)
        llm_filters = metadata.model_dump(exclude_none=True)
    except Exception:  # noqa: BLE001
        llm_filters = {}

    deterministic_filters = {}
    if preprocessed.get("canonical_company"):
        deterministic_filters["company_name"] = preprocessed["canonical_company"]
    if preprocessed.get("fiscal_year"):
        deterministic_filters["fiscal_year"] = preprocessed["fiscal_year"]
    if preprocessed.get("fiscal_quarter"):
        deterministic_filters["fiscal_quarter"] = preprocessed["fiscal_quarter"]

    merged = {**llm_filters, **deterministic_filters}
    return merged


def build_qdrant_filter(filters: dict):
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        conditions.append(
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
        )
    if not conditions:
        return None
    return Filter(must=conditions)


def hybrid_search(query: str, k: int = 20, filters: dict | None = None):
    filters = filters if filters is not None else extract_filters(query)
    qdrant_filter = build_qdrant_filter(filters)
    return get_vector_store().similarity_search(query=query, k=k, filter=qdrant_filter)


def rewrite_query_for_rag_llm(query: str) -> list[str]:
    preprocessed = preprocess_query(query)
    semantic = semantic_enrich_query(query, preprocessed)
    prompt = f"""
You are rewriting a finance-document retrieval query for RAG over English SEC-style filings.
Generate at most 2 short retrieval queries.

Requirements:
- Keep the meaning of the user's request
- Expand with finance terminology when useful
- Prefer wording that is likely to appear in 10-K / 10-Q documents
- Output JSON only in the form: {{"queries": ["...", "..."]}}

User query: {query}
Known normalization:
- company={preprocessed.get("canonical_company")}
- ticker={preprocessed.get("ticker")}
- fiscal_year={preprocessed.get("fiscal_year")}
- fiscal_quarter={preprocessed.get("fiscal_quarter")}
- translated_terms={preprocessed.get("translated_terms")}
- english_query={semantic.get("english_query")}
- semantic_search_queries={semantic.get("search_queries")}
""".strip()
    try:
        response = _quiet_call(get_llm().invoke, prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        data = json.loads(str(content))
        queries = data.get("queries", [])
        if not isinstance(queries, list):
            return []
        cleaned = []
        seen = set()
        for item in queries[:2]:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned
    except Exception:  # noqa: BLE001
        return []


def rewrite_query_for_rag(query: str) -> list[str]:
    preprocessed = preprocess_query(query)
    semantic = semantic_enrich_query(query, preprocessed)
    retrieval_profile = _infer_retrieval_profile(preprocessed, semantic)
    if retrieval_profile in {"live_market", "non_finance"}:
        return []
    rule_queries = build_retrieval_queries(preprocessed)
    llm_queries = rewrite_query_for_rag_llm(query)
    semantic_queries = semantic.get("search_queries") or []
    semantic_english_query = semantic.get("english_query") or ""

    return dedupe_queries(
        [
            preprocessed.get("report_query_en", ""),
            preprocessed.get("english_core_query", ""),
            semantic_english_query,
            *semantic_queries,
            *rule_queries,
            *llm_queries,
        ]
    )


def _infer_retrieval_profile(preprocessed: dict, semantic: dict) -> str:
    semantic_intent = str(semantic.get("intent_hint") or "").strip().lower()
    if semantic_intent in {"live_market", "non_finance"}:
        return semantic_intent
    if semantic_intent in {"historical_rag", "hybrid", "finance_general"}:
        return semantic_intent

    text = (preprocessed.get("normalized_query") or "").lower()
    live_hits = [
        "最新",
        "最近",
        "实时",
        "今天",
        "当前",
        "today",
        "latest",
        "current",
        "stock price",
        "price",
        "股价",
        "走势",
        "trend",
    ]
    historical_hits = [
        "财报",
        "年报",
        "季报",
        "10-k",
        "10-q",
        "annual report",
        "quarterly report",
        "revenue",
        "net income",
        "cash flow",
        "margin",
        "guidance",
        "risk",
        "利润",
        "营收",
        "现金流",
        "财务",
    ]
    live_score = sum(1 for token in live_hits if token in text)
    historical_score = sum(1 for token in historical_hits if token in text)

    if preprocessed.get("live_period"):
        live_score += 2
    if preprocessed.get("fiscal_year") or preprocessed.get("fiscal_quarter"):
        historical_score += 2

    if historical_score > 0 and live_score > 0:
        return "hybrid"
    if historical_score > 0:
        return "historical_rag"
    if live_score > 0:
        return "live_market"
    if preprocessed.get("canonical_company") or preprocessed.get("ticker"):
        return "finance_general"
    return "non_finance"


def multi_query_retrieve(query: str, per_query_k: int = 12) -> list:
    filters = extract_filters(query)
    preprocessed = preprocess_query(query)
    retrieval_queries = _select_retrieval_queries(query)
    candidates = []
    seen = set()

    for query_rank, rewritten in enumerate(retrieval_queries):
        docs = hybrid_search(rewritten, k=per_query_k, filters=filters)
        for doc_rank, doc in enumerate(docs):
            key = (
                doc.metadata.get("source_file"),
                doc.metadata.get("page"),
                doc.metadata.get("chunk_index"),
                doc.metadata.get("table_id"),
                doc.page_content[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append((query_rank, doc_rank, doc))

    return _diversify_candidate_pool(query, preprocessed, candidates)


def _select_retrieval_queries(query: str, limit: int = 6) -> list[str]:
    retrieval_queries = rewrite_query_for_rag(query)
    if not retrieval_queries:
        return []

    preprocessed = preprocess_query(query)
    priority_terms = _build_query_priority_terms(query, preprocessed)
    ranked = sorted(
        enumerate(retrieval_queries),
        key=lambda item: (
            _score_retrieval_query(item[1], priority_terms, preprocessed),
            -item[0],
        ),
        reverse=True,
    )
    return [text for _, text in ranked[:limit]]


def _build_query_priority_terms(query: str, preprocessed: dict) -> list[str]:
    lowered_query = (query or "").lower()
    priority_terms = []

    translated_terms = [str(item).lower() for item in (preprocessed.get("translated_terms") or [])]
    requested_terms = set(translated_terms)
    if "风险" in query or "risk" in lowered_query:
        requested_terms.add("risk")
    if "现金流" in query or "cash flow" in lowered_query:
        requested_terms.add("cash flow")
    if "毛利" in query or "gross margin" in lowered_query or "margin" in lowered_query:
        requested_terms.add("margin")
    if "营收" in query or "收入" in query or "revenue" in lowered_query or "sales" in lowered_query:
        requested_terms.add("revenue")
    if "利润" in query or "profit" in lowered_query or "net income" in lowered_query:
        requested_terms.add("profit")

    for requested in requested_terms:
        priority_terms.extend(RETRIEVAL_SPECIALTY_TERMS.get(requested, []))

    return dedupe_queries([*priority_terms, *RETRIEVAL_DOCUMENT_SIGNALS])


def _score_retrieval_query(text: str, priority_terms: list[str], preprocessed: dict) -> int:
    lowered = (text or "").lower()
    if not lowered:
        return -10_000

    score = 0
    company = str(preprocessed.get("canonical_company") or "").lower()
    ticker = str(preprocessed.get("ticker") or "").lower()
    fiscal_year = str(preprocessed.get("fiscal_year") or "").strip()

    for term in priority_terms:
        if term in lowered:
            score += 5 if term in RETRIEVAL_DOCUMENT_SIGNALS else 8

    if company and company in lowered:
        score += 3
    if ticker and ticker in lowered:
        score += 2
    if fiscal_year and fiscal_year in lowered:
        score += 2

    if _looks_like_document_query(lowered):
        score += 4

    if "|" in lowered:
        score -= 5
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        score -= 3
    if len(text) > 120:
        score -= 2

    return score


def _diversify_candidate_pool(query: str, preprocessed: dict, candidates: list[tuple[int, int, object]], limit: int = 24) -> list:
    if not candidates:
        return []

    preferred_roles = _build_candidate_role_preferences(query, preprocessed)
    ranked = sorted(
        candidates,
        key=lambda item: _score_candidate(item[2], item[0], item[1], preferred_roles),
        reverse=True,
    )

    selected = []
    page_counts: dict[tuple[str | None, int | None], int] = {}
    role_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for _, _, doc in ranked:
        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page")
        source = metadata.get("source_file")
        role = str(metadata.get("page_role") or "general")
        page_key = (str(source), page)

        max_per_page = 1 if metadata.get("is_noise_prone") else 2
        max_per_role = _max_docs_for_role(role, preferred_roles)
        max_per_source = 12 if source and str(source).endswith(".md") else 8

        if page_counts.get(page_key, 0) >= max_per_page:
            continue
        if role_counts.get(role, 0) >= max_per_role:
            continue
        if source_counts.get(str(source), 0) >= max_per_source:
            continue

        selected.append(doc)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        source_counts[str(source)] = source_counts.get(str(source), 0) + 1

        if len(selected) >= limit:
            break

    if selected:
        return selected
    return [doc for _, _, doc in ranked[:limit]]


def _build_candidate_role_preferences(query: str, preprocessed: dict) -> set[str]:
    lowered_query = (query or "").lower()
    requested_terms = {str(item).lower() for item in (preprocessed.get("translated_terms") or [])}

    if "风险" in query or "risk" in lowered_query:
        return {"risk_factors", "md_and_a"}
    if "现金流" in query or "cash flow" in lowered_query:
        return {"cash_flow_statement", "md_and_a"}
    if "毛利" in query or "gross margin" in lowered_query or "margin" in lowered_query:
        return {"gross_margin_discussion", "income_statement", "md_and_a"}
    if (
        "营收" in query
        or "收入" in query
        or "revenue" in lowered_query
        or "sales" in lowered_query
        or "revenue" in requested_terms
    ):
        return {"income_statement", "gross_margin_discussion", "md_and_a"}
    if (
        "利润" in query
        or "profit" in lowered_query
        or "net income" in lowered_query
        or "profit" in requested_terms
    ):
        return {"income_statement", "md_and_a"}
    return {"income_statement", "balance_sheet", "cash_flow_statement", "md_and_a"}


def _score_candidate(doc, query_rank: int, doc_rank: int, preferred_roles: set[str]) -> float:
    metadata = getattr(doc, "metadata", {}) or {}
    role = str(metadata.get("page_role") or "general")
    content_type = str(metadata.get("content_type") or "")
    score = 100.0 - (query_rank * 10.0) - doc_rank

    if role in preferred_roles:
        score += 18.0
    elif role in STRUCTURED_PAGE_ROLES:
        score += 6.0

    if content_type == "table":
        score += 4.0
    if metadata.get("is_noise_prone"):
        score -= 18.0
    if role == "financial_index":
        score -= 10.0
    elif role in {"table_of_contents", "audit_report", "internal_control", "certification"}:
        score -= 12.0

    return score


def _max_docs_for_role(role: str, preferred_roles: set[str]) -> int:
    if role in {"table_of_contents", "audit_report", "internal_control", "certification", "website_disclosure", "exhibit_index"}:
        return 0
    if role == "financial_index":
        return 1
    if role in preferred_roles:
        return 4
    if role in STRUCTURED_PAGE_ROLES:
        return 3
    if role == "general":
        return 8
    return 2


def rerank_results(query: str, documents: list, top_k: int = 5, rerank_query: str | None = None):
    if not documents:
        return []
    documents = documents[:50]
    effective_query = rerank_query or _build_rerank_query(query)
    rerank_profile = _build_rerank_profile(query)
    pairs = [(effective_query, doc.page_content[:2000]) for doc in documents]
    try:
        reranker = _quiet_call(get_reranker)
        scores = _quiet_call(reranker.score, pairs)
        adjusted = []
        for base_score, doc in zip(scores, documents):
            adjusted_score = float(base_score) + _heuristic_rerank_boost(doc, rerank_profile)
            adjusted.append((adjusted_score, doc))
        reranked = sorted(adjusted, key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked[:top_k]]
    except Exception:  # noqa: BLE001
        return documents[:top_k]


def deep_rag_retrieve(query: str) -> str:
    search_results = multi_query_retrieve(query, per_query_k=10)
    if not search_results:
        return ""
    reranked = rerank_results(query, search_results, top_k=5)
    blocks = []
    for doc in reranked:
        meta = doc.metadata
        block = (
            f"[Company: {meta.get('company_name')} | "
            f"Type: {meta.get('doc_type')} | "
            f"Year: {meta.get('fiscal_year')} | "
            f"Page: {meta.get('page')}]\n\n{doc.page_content}"
        )
        blocks.append(block.strip())
    return "\n\n---\n\n".join(blocks)


def deep_rag_search(query: str) -> str:
    """Search through financial documents with hybrid retrieval and rerank."""
    if not query or not query.strip():
        return "Empty query provided."
    try:
        context = deep_rag_retrieve(query)
        if not context:
            return "No relevant information found in the financial documents."
        if len(context) > 8000:
            context = context[:8000] + "\n\n...[truncated]"
        return _build_rag_evidence_response(query, context)
    except Exception as exc:  # noqa: BLE001
        return f"Error during retrieval: {exc}"


def get_deep_rag_search_tool():
    assert_runtime_ready(
        stage="retrieval.get_deep_rag_search_tool",
        packages=["langchain_core"],
    )
    from langchain_core.tools import tool

    return tool(deep_rag_search)


def _build_rag_evidence_response(query: str, context: str) -> str:
    preprocessed = preprocess_query(query)
    requested_terms = _extract_requested_evidence_terms(query, preprocessed)
    found_terms = []
    missing_terms = []
    lowered_context = context.lower()
    for label, variants in requested_terms:
        if any(variant in lowered_context for variant in variants):
            found_terms.append(label)
        else:
            missing_terms.append(label)

    header_lines = [
        "[RAG Evidence Notice]",
        "- The following content is retrieved evidence snippets, not a final verified answer.",
        "- Only make claims that are explicitly supported by the snippets.",
        "- If a requested metric is not explicitly present in the snippets, say the evidence is insufficient.",
    ]
    if found_terms:
        header_lines.append("- Covered evidence terms: " + ", ".join(found_terms))
    if missing_terms:
        header_lines.append("- Potentially missing evidence terms: " + ", ".join(missing_terms))
    return "\n".join(header_lines) + "\n\n[Retrieved Evidence]\n" + context


def _extract_requested_evidence_terms(query: str, preprocessed: dict) -> list[tuple[str, list[str]]]:
    lowered_query = (query or "").lower()
    requested: list[tuple[str, list[str]]] = []
    term_map = {
        "revenue": ["revenue", "net sales", "sales"],
        "net income": ["net income", "profit", "income"],
        "profit": ["profit", "net income", "income from operations"],
        "cash flow": ["cash flow", "operating cash flow", "cash generated"],
        "margin": ["margin", "gross margin", "operating margin"],
        "guidance": ["guidance", "outlook"],
        "risk": ["risk", "risk factors"],
    }

    translated_terms = preprocessed.get("translated_terms") or []
    for term in translated_terms:
        variants = term_map.get(term.lower())
        if variants and (term, variants) not in requested:
            requested.append((term, variants))

    if "净利润" in query and ("net income", term_map["net income"]) not in requested:
        requested.append(("net income", term_map["net income"]))
    if "利润" in query and ("profit", term_map["profit"]) not in requested:
        requested.append(("profit", term_map["profit"]))
    if "营收" in query or "收入" in query:
        if ("revenue", term_map["revenue"]) not in requested:
            requested.append(("revenue", term_map["revenue"]))

    for label, variants in term_map.items():
        if any(variant in lowered_query for variant in variants) and (label, variants) not in requested:
            requested.append((label, variants))

    return requested[:6]


def _build_rerank_query(query: str) -> str:
    preprocessed = preprocess_query(query)
    semantic = semantic_enrich_query(query, preprocessed)
    retrieval_queries = rewrite_query_for_rag(query)
    return (
        preprocessed.get("report_query_en")
        or next((item for item in retrieval_queries if _looks_like_document_query(item)), "")
        or semantic.get("english_query")
        or preprocessed.get("english_core_query")
        or query
    )


def _looks_like_document_query(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    document_signals = [
        "10-k",
        "10-q",
        "annual report",
        "quarterly report",
        "financial report",
        "revenue",
        "net income",
        "cash flow",
        "margin",
    ]
    return any(token in lowered for token in document_signals)


def _build_rerank_profile(query: str) -> dict:
    preprocessed = preprocess_query(query)
    lowered_query = (query or "").lower()
    translated_terms = {str(item).lower() for item in (preprocessed.get("translated_terms") or [])}

    focus = "general"
    if "风险" in query or "risk" in lowered_query or "risk" in translated_terms:
        focus = "risk"
    elif "现金流" in query or "cash flow" in lowered_query or "cash flow" in translated_terms:
        focus = "cash_flow"
    elif "毛利" in query or "gross margin" in lowered_query or "margin" in translated_terms:
        focus = "margin"
    elif (
        "营收" in query
        or "收入" in query
        or "revenue" in lowered_query
        or "sales" in lowered_query
        or "revenue" in translated_terms
    ):
        focus = "revenue"
    elif (
        "利润" in query
        or "profit" in lowered_query
        or "net income" in lowered_query
        or "profit" in translated_terms
    ):
        focus = "profit"

    return {"focus": focus}


def _heuristic_rerank_boost(doc, rerank_profile: dict | None = None) -> float:
    metadata = getattr(doc, "metadata", {}) or {}
    content = (getattr(doc, "page_content", "") or "").lower()
    normalized_content = " ".join(content.split())
    score = 0.0
    focus = str((rerank_profile or {}).get("focus") or "general")
    role = str(metadata.get("page_role") or "general")

    if metadata.get("content_type") == "table":
        score += 1.2

    positive_hits = sum(
        1 for term in FINANCIAL_STRUCTURE_POSITIVE_TERMS if term in normalized_content
    )
    negative_hits = sum(
        1 for term in FINANCIAL_STRUCTURE_NEGATIVE_TERMS if term in normalized_content
    )
    score += positive_hits * 0.35
    score -= negative_hits * 0.45

    certification_patterns = [
        "i, timothy d. cook, certify",
        "i have reviewed this annual report",
        "fairly presents in all material respects",
    ]
    if any(pattern in normalized_content for pattern in certification_patterns):
        score -= 1.0

    if "internal control over financial reporting" in normalized_content:
        score -= 1.1

    if "incorporated by reference" in normalized_content or "form s-8" in normalized_content:
        score -= 0.9

    if (
        "apple inc." in normalized_content
        and "form 10-k" in normalized_content
        and "table of contents" in normalized_content
    ):
        score -= 0.8

    role_preferences = RERANK_ROLE_PREFERENCES.get(focus, {})
    role_penalties = RERANK_ROLE_PENALTIES.get(focus, {})
    score += role_preferences.get(role, 0.0)
    score += role_penalties.get(role, 0.0)

    for term in RERANK_CONTENT_PREFERENCES.get(focus, []):
        if term in normalized_content:
            score += 0.45
    for term in RERANK_CONTENT_PENALTIES.get(focus, []):
        if term in normalized_content:
            score -= 0.55

    return score


def _quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
        warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests.*")
        with redirect_stdout(sink), redirect_stderr(sink):
            return func(*args, **kwargs)
