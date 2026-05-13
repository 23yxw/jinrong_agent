"""Structured dialog state manager for finance task continuity across turns."""

from __future__ import annotations

import json
from pathlib import Path
import re

from config import DATA_DIR
from query_preprocessor import COMPANY_ALIAS_TO_CANONICAL, CANONICAL_TO_TICKER, preprocess_query

STATE_DIR = DATA_DIR / "dialog_state"

FOLLOW_UP_MARKERS = [
    "那",
    "那么",
    "然后",
    "接着",
    "继续",
    "再",
    "另外",
    "呢",
    "and what about",
    "what about",
    "how about",
]

CANCEL_COMPARE_MARKERS = [
    "先不比",
    "不比较",
    "取消比较",
    "不用比较",
    "不要比较",
    "别比了",
    "先不要对比",
    "stop comparing",
    "don't compare",
    "no comparison",
]

REFERENCE_MARKERS = [
    "它",
    "这家",
    "该公司",
    "这家公司",
    "这只股票",
    "这个公司",
    "the company",
    "the stock",
    "it",
]

COMPARE_MARKERS = [
    "对比",
    "比较",
    "和",
    "相比",
    "vs",
    "versus",
    "compare",
]

DOCUMENT_FOCUS_HINTS = [
    "财报",
    "营收",
    "收入",
    "利润",
    "现金流",
    "风险",
    "指引",
    "margin",
    "revenue",
    "profit",
    "cash flow",
    "guidance",
    "risk",
]


def load_dialog_state(thread_id: str) -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(thread_id)
    if not path.exists():
        return _default_state()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return _default_state()


def save_dialog_state(thread_id: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(thread_id).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_dialog_state(thread_id: str) -> bool:
    path = _state_path(thread_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def resolve_query_with_state(thread_id: str, query: str) -> dict:
    state = load_dialog_state(thread_id)
    raw_query = query or ""
    current_pre = preprocess_query(raw_query)
    companies_in_query = _extract_companies(raw_query)
    compare_mode = _has_compare_markers(raw_query)
    follow_up = _looks_like_follow_up(raw_query)
    cancel_compare = _wants_to_cancel_comparison(raw_query)

    context_parts: list[str] = []
    state_used = False
    trace_reasons: list[str] = []

    if compare_mode and state.get("primary_company"):
        candidates = [item for item in companies_in_query if item != state["primary_company"]]
        if candidates and state["primary_company"] not in companies_in_query:
            state_used = True
            base_company = state["primary_company"]
            context_parts.append(f"compare {base_company} with {candidates[0]}")
            if state.get("primary_ticker"):
                context_parts.append(f"base ticker {state['primary_ticker']}")
            trace_reasons.append("inherit_primary_company_for_comparison")
    elif (
        not current_pre.get("canonical_company")
        and state.get("primary_company")
        and (follow_up or _refers_to_previous_entity(raw_query) or len(raw_query.strip()) <= 18)
    ):
        state_used = True
        context_parts.append(f"company {state['primary_company']}")
        if state.get("primary_ticker"):
            context_parts.append(f"ticker {state['primary_ticker']}")
        trace_reasons.append("inherit_primary_company_for_follow_up")

    if (
        not current_pre.get("fiscal_year")
        and not current_pre.get("live_period")
        and state.get("fiscal_year")
        and _looks_like_document_follow_up(raw_query)
    ):
        state_used = True
        context_parts.append(f"fiscal year {state['fiscal_year']}")
        trace_reasons.append("inherit_fiscal_year_for_document_follow_up")

    if (
        not current_pre.get("translated_terms")
        and state.get("analysis_focus")
        and _looks_like_focus_follow_up(raw_query)
    ):
        state_used = True
        context_parts.append("focus " + ", ".join(state["analysis_focus"][:3]))
        trace_reasons.append("inherit_analysis_focus_for_follow_up")

    if cancel_compare and state.get("comparison_company"):
        trace_reasons.append("clear_previous_comparison_context")

    effective_query = raw_query
    if context_parts:
        effective_query = f"{raw_query} [dialog context: {'; '.join(context_parts)}]"
    resolved_pre = preprocess_query(effective_query)
    note = _build_dialog_state_note(state, raw_query, effective_query, state_used)
    trace = build_dialog_state_trace(
        state=state,
        raw_query=raw_query,
        effective_query=effective_query,
        preprocessed=resolved_pre,
        state_used=state_used,
        reasons=trace_reasons,
        compare_mode=compare_mode,
        cancel_compare=cancel_compare,
    )

    return {
        "raw_query": raw_query,
        "effective_query": effective_query,
        "state_used": state_used,
        "dialog_state": state,
        "dialog_state_note": note,
        "dialog_state_trace": trace,
        "preprocessed": resolved_pre,
    }


def update_dialog_state(
    thread_id: str,
    *,
    original_query: str,
    effective_query: str,
    preprocessed: dict,
    route: dict,
) -> dict:
    state = load_dialog_state(thread_id)
    companies = _extract_companies(effective_query)
    compare_mode = _has_compare_markers(original_query)
    cancel_compare = _wants_to_cancel_comparison(original_query)

    if preprocessed.get("canonical_company"):
        if not (compare_mode and state.get("primary_company") and state["primary_company"] in companies):
            state["primary_company"] = preprocessed.get("canonical_company")
            state["primary_ticker"] = preprocessed.get("ticker") or CANONICAL_TO_TICKER.get(
                preprocessed.get("canonical_company")
            )

    if cancel_compare:
        state["comparison_company"] = None
        state["comparison_ticker"] = None
    elif compare_mode and state.get("primary_company"):
        others = [item for item in companies if item != state["primary_company"]]
        if others:
            state["comparison_company"] = others[0]
            state["comparison_ticker"] = CANONICAL_TO_TICKER.get(others[0])
    elif companies:
        state["comparison_company"] = None
        state["comparison_ticker"] = None

    if preprocessed.get("fiscal_year"):
        state["fiscal_year"] = preprocessed["fiscal_year"]
    if preprocessed.get("fiscal_quarter"):
        state["fiscal_quarter"] = preprocessed["fiscal_quarter"]
    if preprocessed.get("live_period"):
        state["live_period"] = preprocessed["live_period"]
    if preprocessed.get("live_interval"):
        state["live_interval"] = preprocessed["live_interval"]

    if preprocessed.get("translated_terms"):
        state["analysis_focus"] = _merge_focus(
            state.get("analysis_focus", []),
            preprocessed.get("translated_terms") or [],
        )

    state["last_intent"] = route.get("intent")
    state["last_query"] = original_query
    state["last_effective_query"] = effective_query

    save_dialog_state(thread_id, state)
    return state


def build_dialog_state_trace(
    *,
    state: dict,
    raw_query: str,
    effective_query: str,
    preprocessed: dict,
    state_used: bool,
    reasons: list[str] | None = None,
    compare_mode: bool = False,
    cancel_compare: bool = False,
) -> dict:
    return {
        "raw_query": raw_query,
        "effective_query": effective_query,
        "state_used": state_used,
        "reasons": list(reasons or []),
        "compare_mode": compare_mode,
        "cancel_compare": cancel_compare,
        "resolved_company": preprocessed.get("canonical_company"),
        "resolved_ticker": preprocessed.get("ticker"),
        "resolved_fiscal_year": preprocessed.get("fiscal_year"),
        "resolved_live_period": preprocessed.get("live_period"),
        "state_snapshot": {
            "primary_company": state.get("primary_company"),
            "primary_ticker": state.get("primary_ticker"),
            "comparison_company": state.get("comparison_company"),
            "comparison_ticker": state.get("comparison_ticker"),
            "fiscal_year": state.get("fiscal_year"),
            "fiscal_quarter": state.get("fiscal_quarter"),
            "live_period": state.get("live_period"),
            "analysis_focus": list(state.get("analysis_focus") or []),
            "last_intent": state.get("last_intent"),
        },
    }


def format_dialog_state_trace(trace: dict) -> str:
    if not trace:
        return ""
    lines = [
        "[dialog_state] "
        f"used={trace.get('state_used')} "
        f"company={trace.get('resolved_company')} "
        f"ticker={trace.get('resolved_ticker')} "
        f"year={trace.get('resolved_fiscal_year')} "
        f"period={trace.get('resolved_live_period')}"
    ]
    reasons = trace.get("reasons") or []
    if reasons:
        lines.append("[dialog_state] reasons=" + ",".join(reasons))
    snapshot = trace.get("state_snapshot") or {}
    snapshot_pairs = []
    for key in [
        "primary_company",
        "comparison_company",
        "fiscal_year",
        "live_period",
        "last_intent",
    ]:
        value = snapshot.get(key)
        if value:
            snapshot_pairs.append(f"{key}={value}")
    if snapshot_pairs:
        lines.append("[dialog_state] snapshot=" + " ".join(snapshot_pairs))
    if trace.get("effective_query") and trace.get("effective_query") != trace.get("raw_query"):
        lines.append("[dialog_state] effective_query=" + str(trace.get("effective_query")))
    return "\n".join(lines)


def build_memory_priority_note(conflicts: list[str] | None = None) -> str:
    note = (
        "If conversation memory summary or older recent messages conflict with the structured "
        "dialog state or the resolved_query, prioritize the structured dialog state and resolved_query. "
        "Treat adaptive memory as secondary historical context."
    )
    if conflicts:
        note += "\nDetected memory/state conflicts:\n" + "\n".join(f"- {item}" for item in conflicts[:5])
    return note


def build_memory_conflict_event(
    *,
    thread_id: str,
    conflicts: list[str],
    dialog_state: dict | None = None,
    memory_session: dict | None = None,
    effective_query: str | None = None,
) -> dict:
    state = dialog_state or {}
    session = memory_session or {}
    recent_messages = session.get("recent_messages", []) or []
    return {
        "event": "memory_state_conflict",
        "thread_id": thread_id,
        "effective_query": effective_query,
        "conflict_count": len(conflicts),
        "conflicts": list(conflicts),
        "dialog_state": {
            "primary_company": state.get("primary_company"),
            "primary_ticker": state.get("primary_ticker"),
            "comparison_company": state.get("comparison_company"),
            "comparison_ticker": state.get("comparison_ticker"),
            "fiscal_year": state.get("fiscal_year"),
            "live_period": state.get("live_period"),
            "analysis_focus": list(state.get("analysis_focus") or []),
            "last_intent": state.get("last_intent"),
        },
        "adaptive_memory": {
            "summary_chars": len(str(session.get("summary", ""))),
            "recent_count": len(recent_messages),
        },
    }


def format_memory_conflict_event(event: dict) -> str:
    return "[memory_event] " + json.dumps(event, ensure_ascii=False, sort_keys=True)


def detect_memory_state_conflicts(memory_session: dict, dialog_state: dict) -> list[str]:
    conflicts: list[str] = []
    state = dialog_state or {}
    session = memory_session or {}
    all_text = "\n".join(
        [
            str(session.get("summary", "")),
            "\n".join(str(item.get("content", "")) for item in session.get("recent_messages", [])),
        ]
    ).strip()
    if not all_text:
        return conflicts

    mentioned_companies = _extract_companies(all_text)
    primary_company = state.get("primary_company")
    comparison_company = state.get("comparison_company")
    if primary_company and mentioned_companies and primary_company not in mentioned_companies:
        conflicts.append(
            f"dialog_state.primary_company={primary_company} missing from adaptive_memory companies={mentioned_companies}"
        )
    allowed_companies = {item for item in [primary_company, comparison_company] if item}
    if primary_company and mentioned_companies:
        unexpected_companies = [item for item in mentioned_companies if item not in allowed_companies]
        if unexpected_companies and not comparison_company:
            conflicts.append(
                "adaptive_memory contains stale company mentions not reflected in dialog_state: "
                f"{unexpected_companies}"
            )
    if comparison_company and comparison_company not in mentioned_companies:
        conflicts.append(
            f"dialog_state.comparison_company={comparison_company} missing from adaptive_memory companies={mentioned_companies}"
        )

    primary_ticker = state.get("primary_ticker")
    if primary_ticker and primary_ticker.lower() not in all_text.lower() and primary_company in mentioned_companies:
        conflicts.append(
            f"dialog_state.primary_ticker={primary_ticker} missing from adaptive_memory text while company is present"
        )

    live_period = state.get("live_period")
    if live_period and live_period not in all_text and "stock" in all_text.lower():
        conflicts.append(
            f"dialog_state.live_period={live_period} missing from adaptive_memory stock discussion"
        )

    return conflicts


def _default_state() -> dict:
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


def _build_dialog_state_note(state: dict, raw_query: str, effective_query: str, state_used: bool) -> str:
    if not state_used:
        return ""
    fields = [
        f"raw_query={raw_query}",
        f"resolved_query={effective_query}",
    ]
    for key in [
        "primary_company",
        "primary_ticker",
        "comparison_company",
        "comparison_ticker",
        "fiscal_year",
        "fiscal_quarter",
        "live_period",
        "live_interval",
        "last_intent",
    ]:
        value = state.get(key)
        if value:
            fields.append(f"{key}={value}")
    if state.get("analysis_focus"):
        fields.append("analysis_focus=" + ",".join(state["analysis_focus"][:4]))
    return (
        "Structured dialog state inherited from previous turns. "
        "Use resolved_query for routing and tool planning when helpful.\n"
        + "\n".join(f"- {item}" for item in fields)
    )


def _extract_companies(text: str) -> list[str]:
    normalized = (text or "").lower()
    found = []
    aliases = sorted(COMPANY_ALIAS_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, canonical in aliases:
        if alias in normalized and canonical not in found:
            found.append(canonical)
    return found


def _merge_focus(existing: list[str], new_items: list[str]) -> list[str]:
    merged = list(existing or [])
    for item in new_items or []:
        cleaned = str(item).strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged[:6]


def _looks_like_follow_up(query: str) -> bool:
    lowered = (query or "").lower().strip()
    if not lowered:
        return False
    if any(marker in lowered for marker in FOLLOW_UP_MARKERS):
        return True
    return len(lowered) <= 12


def _refers_to_previous_entity(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in REFERENCE_MARKERS)


def _has_compare_markers(query: str) -> bool:
    lowered = (query or "").lower()
    explicit_markers = [marker for marker in COMPARE_MARKERS if marker != "和"]
    if any(marker in lowered for marker in explicit_markers):
        return True
    if re.search(r"和.{0,12}比", query):
        return True
    companies = _extract_companies(query)
    if len(companies) >= 2 and "和" in query:
        return True
    return False


def _looks_like_document_follow_up(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in DOCUMENT_FOCUS_HINTS)


def _looks_like_focus_follow_up(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in ["只看", "重点", "怎么看", "聚焦", "focus"])


def _wants_to_cancel_comparison(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in CANCEL_COMPARE_MARKERS)


def _state_path(thread_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", thread_id)
    return STATE_DIR / f"{safe}.json"
