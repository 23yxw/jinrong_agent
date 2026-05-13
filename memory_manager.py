"""Adaptive conversation memory manager with token-aware compression."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from config import DATA_DIR
from model_factory import get_chat_model

MEMORY_DIR = DATA_DIR / "adaptive_memory"
SOFT_TOKEN_LIMIT = int(os.getenv("ADAPTIVE_MEMORY_SOFT_LIMIT", "12000"))
HARD_TOKEN_LIMIT = int(os.getenv("ADAPTIVE_MEMORY_HARD_LIMIT", "16000"))
KEEP_LAST_MESSAGES = int(os.getenv("ADAPTIVE_MEMORY_KEEP_LAST", "8"))
MAX_TOOL_SNIPPET_CHARS = int(os.getenv("ADAPTIVE_MEMORY_TOOL_CHARS", "600"))


def load_session(thread_id: str) -> dict:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(thread_id)
    if not path.exists():
        return {"summary": "", "recent_messages": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"summary": "", "recent_messages": []}


def save_session(thread_id: str, session: dict) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _session_path(thread_id).write_text(
        json.dumps(session, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_session(thread_id: str) -> bool:
    path = _session_path(thread_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def build_input_messages(thread_id: str, query: str):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    session = load_session(thread_id)
    messages = []

    summary = session.get("summary", "").strip()
    if summary:
        messages.append(
            SystemMessage(
                content=(
                    "Conversation memory summary from previous rounds. "
                    "Treat this as compressed historical context:\n\n"
                    f"{summary}"
                )
            )
        )

    for item in session.get("recent_messages", []):
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            messages.append(SystemMessage(content=f"Previous tool result:\n{content}"))

    messages.append(HumanMessage(content=query))
    return messages


def record_interaction(
    thread_id: str,
    user_query: str,
    assistant_response: str,
    tool_outputs: list[str] | None = None,
) -> dict:
    session = load_session(thread_id)
    recent = session.setdefault("recent_messages", [])
    recent.append({"role": "user", "content": user_query})

    for tool_output in tool_outputs or []:
        cleaned = _clean_tool_output(tool_output)
        if cleaned:
            recent.append({"role": "tool", "content": cleaned})

    if assistant_response.strip():
        recent.append({"role": "assistant", "content": assistant_response.strip()})

    info = maybe_compact_session(session)
    save_session(thread_id, session)
    return info


def maybe_compact_session(session: dict) -> dict:
    summary = session.get("summary", "")
    recent = session.get("recent_messages", [])
    estimated_tokens = estimate_session_tokens(session)
    if estimated_tokens <= SOFT_TOKEN_LIMIT:
        return {
            "compacted": False,
            "estimated_tokens": estimated_tokens,
            "summary_chars": len(summary),
            "recent_count": len(recent),
        }

    keep_count = min(KEEP_LAST_MESSAGES, len(recent))
    old_messages = recent[:-keep_count] if keep_count else recent[:]
    kept_messages = recent[-keep_count:] if keep_count else []
    if not old_messages:
        trimmed = _trim_recent_messages_to_hard_limit(recent)
        session["recent_messages"] = trimmed
        return {
            "compacted": False,
            "estimated_tokens": estimate_session_tokens(session),
            "summary_chars": len(summary),
            "recent_count": len(trimmed),
        }

    old_text = _messages_to_text(old_messages)
    new_summary = _summarize_history(summary, old_text)
    session["summary"] = new_summary
    session["recent_messages"] = kept_messages

    if estimate_session_tokens(session) > HARD_TOKEN_LIMIT:
        session["recent_messages"] = _trim_recent_messages_to_hard_limit(
            session["recent_messages"]
        )

    return {
        "compacted": True,
        "estimated_tokens": estimate_session_tokens(session),
        "summary_chars": len(session["summary"]),
        "recent_count": len(session["recent_messages"]),
    }


def estimate_session_tokens(session: dict) -> int:
    text = session.get("summary", "") + "\n" + _messages_to_text(
        session.get("recent_messages", [])
    )
    return estimate_text_tokens(text)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _summarize_history(existing_summary: str, old_text: str) -> str:
    model = get_chat_model(temperature=0)
    prompt = (
        "You are compressing a long-running finance-agent session.\n"
        "Preserve only durable information needed for future reasoning.\n"
        "Prefer the latest active company, comparison target, fiscal period, and market timeframe.\n"
        "If older companies or topics were superseded later, mark them as obsolete instead of keeping them active.\n"
        "Output concise markdown with these sections exactly:\n"
        "1. Active Task State\n"
        "   - current primary company / ticker\n"
        "   - current comparison company if any\n"
        "   - active fiscal year / quarter if any\n"
        "   - active market timeframe if any\n"
        "   - active analysis focus\n"
        "2. User Goals\n"
        "3. Confirmed Facts and Figures\n"
        "4. Tool Results and Data Sources\n"
        "5. Errors, Fallbacks, and Caveats\n"
        "6. Obsolete or Superseded Context\n"
        "7. Open Questions / Next Steps\n\n"
        "Do not let obsolete companies remain the active subject.\n"
        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        f"Older conversation to compress:\n{old_text}\n"
    )
    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    return str(content).strip()


def _messages_to_text(messages: list[dict]) -> str:
    lines = []
    for item in messages:
        role = item.get("role", "unknown")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def _trim_recent_messages_to_hard_limit(messages: list[dict]) -> list[dict]:
    trimmed = list(messages)
    while trimmed and estimate_text_tokens(_messages_to_text(trimmed)) > HARD_TOKEN_LIMIT:
        trimmed.pop(0)
    return trimmed


def _clean_tool_output(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:MAX_TOOL_SNIPPET_CHARS]


def _session_path(thread_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", thread_id)
    return MEMORY_DIR / f"{safe}.json"
