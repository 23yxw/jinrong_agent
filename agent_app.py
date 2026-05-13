"""Agent assembly and interactive streaming helpers."""

import json
import time
import warnings

from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

from config import SYSTEM_PROMPT
from dialog_state_manager import (
    build_memory_priority_note,
    build_memory_conflict_event,
    detect_memory_state_conflicts,
    format_dialog_state_trace,
    format_memory_conflict_event,
    load_dialog_state,
    resolve_query_with_state,
    update_dialog_state,
)
from intent_router import build_agent_routing_note, route_query
from memory_manager import build_input_messages, load_session, record_interaction
from mcp_tools import get_live_finance_researcher_tool, live_finance_researcher
from model_factory import get_chat_model
from query_preprocessor import semantic_enrich_query, split_hybrid_query
from retrieval import deep_rag_search, get_deep_rag_search_tool
from runtime_checks import assert_runtime_ready

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
    category=LangChainPendingDeprecationWarning,
)


def create_default_agent(with_middleware: bool = True, enable_memory: bool = True):
    assert_runtime_ready(
        stage="agent_app.create_default_agent",
        packages=["langchain", "langgraph"],
    )
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware

    _ = enable_memory
    model = get_chat_model()

    middleware = []
    if with_middleware:
        middleware = [
            TodoListMiddleware(),
        ]

    agent_kwargs = dict(
        model=model,
        tools=[get_deep_rag_search_tool(), get_live_finance_researcher_tool()],
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
    )
    return create_agent(**agent_kwargs)


def stream_agent_response(
    agent,
    query: str,
    thread_id: str = "default",
    enable_memory: bool = True,
    resolved: dict | None = None,
):
    assert_runtime_ready(
        stage="agent_app.stream_agent_response",
        packages=["langchain_core"],
    )
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    print(f"\nUser: {query}\n")
    print("Agent:\n")
    resolution = resolved or resolve_query_with_state(thread_id, query)
    effective_query = resolution["effective_query"]
    preprocessed = resolution["preprocessed"]
    route = route_query(effective_query, preprocessed)
    dialog_state_note_text = resolution.get("dialog_state_note") or ""
    dialog_state_trace = resolution.get("dialog_state_trace") or {}
    formatted_trace = format_dialog_state_trace(dialog_state_trace)
    if formatted_trace:
        print(formatted_trace)
    memory_conflicts: list[str] = []
    memory_session = {}
    dialog_state = {}
    if enable_memory:
        memory_session = load_session(thread_id)
        dialog_state = load_dialog_state(thread_id)
        memory_conflicts = detect_memory_state_conflicts(
            memory_session,
            dialog_state,
        )
        if memory_conflicts:
            event = build_memory_conflict_event(
                thread_id=thread_id,
                conflicts=memory_conflicts,
                dialog_state=dialog_state,
                memory_session=memory_session,
                effective_query=effective_query,
            )
            print(format_memory_conflict_event(event))
    if route.get("needs_clarification"):
        print(route.get("clarification_question"))
        return
    if route.get("intent") == "non_finance":
        print(
            "抱歉，我当前主要处理金融研究、财报分析、股票行情和投资风险相关问题。"
            "如果你愿意，可以告诉我公司名、股票代码、年份或你关心的财务指标。"
        )
        return
    if route.get("intent") == "hybrid":
        _run_hybrid_workflow(
            query=query,
            thread_id=thread_id,
            enable_memory=enable_memory,
            preprocessed=preprocessed,
            effective_query=effective_query,
            route=route,
        )
        return
    routing_note = SystemMessage(
        content=build_agent_routing_note(effective_query, preprocessed=preprocessed, route=route)
    )
    state_note = SystemMessage(content=dialog_state_note_text) if dialog_state_note_text else None
    memory_priority_note = SystemMessage(
        content=build_memory_priority_note(memory_conflicts)
    )
    if enable_memory:
        input_messages = build_input_messages(thread_id, query)
    else:
        input_messages = [HumanMessage(content=query)]
    if input_messages and isinstance(input_messages[-1], HumanMessage):
        input_messages.insert(len(input_messages) - 1, memory_priority_note)
    else:
        input_messages.insert(0, memory_priority_note)
    if state_note:
        if input_messages and isinstance(input_messages[-1], HumanMessage):
            input_messages.insert(len(input_messages) - 1, state_note)
        else:
            input_messages.insert(0, state_note)
    if input_messages and isinstance(input_messages[-1], HumanMessage):
        input_messages.insert(len(input_messages) - 1, routing_note)
    else:
        input_messages.insert(0, routing_note)

    assistant_chunks: list[str] = []
    tool_outputs: list[str] = []

    try:
        for chunk in agent.stream(
            {"messages": input_messages},
            stream_mode="messages",
        ):
            message = chunk[0] if isinstance(chunk, tuple) else chunk

            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"\nTool Called: {tool_call['name']}")
                    print(f"Args: {tool_call['args']}\n")
            elif isinstance(message, ToolMessage):
                tool_text = str(message.content)
                tool_outputs.append(tool_text)
                print(f"\nTool Output:\n{tool_text[:1000]}\n")
            elif isinstance(message, AIMessage) and message.content:
                text = str(message.content)
                assistant_chunks.append(text)
                print(text, end="", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\nError: {exc}")
        return

    if enable_memory:
        info = record_interaction(
            thread_id=thread_id,
            user_query=query,
            assistant_response="".join(assistant_chunks),
            tool_outputs=tool_outputs,
        )
        update_dialog_state(
            thread_id,
            original_query=query,
            effective_query=effective_query,
            preprocessed=preprocessed,
            route=route,
        )
        status = "yes" if info.get("compacted") else "no"
        print(
            f"\n[memory] compacted={status} "
            f"tokens~{info.get('estimated_tokens')} "
            f"recent={info.get('recent_count')} "
            f"summary_chars={info.get('summary_chars')}"
        )
    else:
        update_dialog_state(
            thread_id,
            original_query=query,
            effective_query=effective_query,
            preprocessed=preprocessed,
            route=route,
        )


def _run_hybrid_workflow(
    *,
    query: str,
    thread_id: str,
    enable_memory: bool,
    preprocessed: dict,
    effective_query: str,
    route: dict,
) -> None:
    semantic = semantic_enrich_query(effective_query, preprocessed)
    split = split_hybrid_query(effective_query, preprocessed=preprocessed, semantic=semantic)
    document_query = split.get("document_query") or query
    market_query = split.get("market_query") or query

    print("Hybrid Planner:\n")
    print(
        json.dumps(
            {
                "document_query": document_query,
                "market_query": market_query,
                "document_focus": split.get("document_focus"),
                "market_focus": split.get("market_focus"),
                "analysis_source": split.get("analysis_source"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    tool_outputs: list[str] = []
    execution_trace = {
        "intent": "hybrid",
        "analysis_source": split.get("analysis_source"),
        "document_query": document_query,
        "market_query": market_query,
    }
    if enable_memory:
        memory_session = load_session(thread_id)
        dialog_state = load_dialog_state(thread_id)
        memory_conflicts = detect_memory_state_conflicts(
            memory_session,
            dialog_state,
        )
        if memory_conflicts:
            execution_trace["memory_conflicts"] = memory_conflicts
            event = build_memory_conflict_event(
                thread_id=thread_id,
                conflicts=memory_conflicts,
                dialog_state=dialog_state,
                memory_session=memory_session,
                effective_query=effective_query,
            )
            print(format_memory_conflict_event(event))

    print("\nTool Called: deep_rag_search")
    print(f"Args: {{'query': {document_query!r}}}\n")
    doc_start = time.perf_counter()
    document_result = deep_rag_search(document_query)
    doc_elapsed = round(time.perf_counter() - doc_start, 3)
    tool_outputs.append(f"[deep_rag_search]\n{document_result}")
    print(f"Tool Output:\n{document_result[:1200]}\n")
    execution_trace["document_result_chars"] = len(document_result or "")
    execution_trace["document_elapsed_seconds"] = doc_elapsed

    print("\nTool Called: live_finance_researcher")
    print(f"Args: {{'query': {market_query!r}}}\n")
    market_start = time.perf_counter()
    market_result = live_finance_researcher(market_query)
    market_elapsed = round(time.perf_counter() - market_start, 3)
    tool_outputs.append(f"[live_finance_researcher]\n{market_result}")
    print(f"Tool Output:\n{market_result[:1200]}\n")
    execution_trace["market_result_chars"] = len(market_result or "")
    execution_trace["market_elapsed_seconds"] = market_elapsed

    print(
        "Hybrid Trace:\n"
        + json.dumps(execution_trace, ensure_ascii=False, indent=2)
        + "\n"
    )

    summary = _synthesize_hybrid_answer(
        query=query,
        document_query=document_query,
        market_query=market_query,
        document_result=document_result,
        market_result=market_result,
    )
    print(summary, end="", flush=True)

    if enable_memory:
        info = record_interaction(
            thread_id=thread_id,
            user_query=query,
            assistant_response=summary,
            tool_outputs=tool_outputs,
        )
        update_dialog_state(
            thread_id,
            original_query=query,
            effective_query=effective_query,
            preprocessed=preprocessed,
            route=route,
        )
        status = "yes" if info.get("compacted") else "no"
        print(
            f"\n[memory] compacted={status} "
            f"tokens~{info.get('estimated_tokens')} "
            f"recent={info.get('recent_count')} "
            f"summary_chars={info.get('summary_chars')}"
        )
    else:
        update_dialog_state(
            thread_id,
            original_query=query,
            effective_query=effective_query,
            preprocessed=preprocessed,
            route=route,
        )


def _synthesize_hybrid_answer(
    *,
    query: str,
    document_query: str,
    market_query: str,
    document_result: str,
    market_result: str,
) -> str:
    model = get_chat_model()
    prompt = f"""
You are synthesizing a hybrid finance answer from two different evidence streams.

Original user question:
{query}

Document-oriented subquery:
{document_query}

Market-oriented subquery:
{market_query}

Document evidence from deep_rag_search:
{document_result}

Live market evidence from live_finance_researcher:
{market_result}

Requirements:
- Answer in Chinese.
- Organize the answer with these sections:
  1. 财报/历史依据
  2. 实时行情依据
  3. 综合判断
  4. 风险与不确定性
- If either evidence stream is weak or missing, say so explicitly.
- Do not fabricate figures not present in the evidence.
- Cite the source type in each section, such as 文档检索 or Yahoo Finance/MCP.
- Treat the tool outputs as the only admissible evidence.
- If the document evidence only contains generic disclosure text, audit boilerplate, or governance language, explicitly mark the document evidence as weak.
- If the market evidence only shows recent price movement, do not use it to claim long-term importance.
- For questions asking "which is more important", only choose one side when the provided evidence clearly supports it.
- If the evidence does not clearly support a ranking, state: "当前证据不足以严格判断哪一个更重要".
- Do not add generic investment principles unless you clearly label them as background rather than evidence-backed conclusions.
""".strip()
    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    return str(content).strip()
