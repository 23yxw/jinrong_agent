"""Entry point for the financial research agent."""

import argparse
import uuid

from dotenv import load_dotenv

from agent_app import create_default_agent, stream_agent_response
from dialog_state_manager import reset_dialog_state, resolve_query_with_state
from intent_router import route_query
from memory_manager import reset_session

load_dotenv()

NON_FINANCE_REPLY = (
    "抱歉，我当前主要处理金融研究、财报分析、股票行情和投资风险相关问题。"
    "如果你愿意，我可以帮助你分析上市公司财报、股价走势、估值、盈利能力或市场风险。"
)


def _build_args():
    parser = argparse.ArgumentParser(description="Run financial research agent.")
    parser.add_argument(
        "--query",
        default="分析一下特斯拉最近的股价趋势，并给出投资建议",
        help="User query for the agent.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id. If omitted, a unique id is generated.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable checkpoint memory and middleware for clean debugging.",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Clear adaptive_memory and dialog_state for the selected thread before running.",
    )
    return parser.parse_args()


def main():
    args = _build_args()
    enable_memory = not args.no_memory
    thread_id = args.thread_id or f"run-{uuid.uuid4().hex[:12]}"
    if args.reset_session:
        memory_reset = reset_session(thread_id)
        state_reset = reset_dialog_state(thread_id)
        print(
            f"[session] reset thread_id={thread_id} "
            f"adaptive_memory={'yes' if memory_reset else 'no'} "
            f"dialog_state={'yes' if state_reset else 'no'}"
        )
    resolution = resolve_query_with_state(thread_id, args.query)
    preprocessed = resolution["preprocessed"]
    route = route_query(resolution["effective_query"], preprocessed)
    agent = create_default_agent(
        with_middleware=enable_memory,
        enable_memory=enable_memory,
    )
    print(f"[agent] thread_id={thread_id} | memory={'on' if enable_memory else 'off'}")
    print(
        "[query] "
        f"state_used={resolution.get('state_used')} "
        f"intent={route.get('intent')} "
        f"rule_intent={route.get('rule_intent')} "
        f"semantic_intent={route.get('semantic_intent')} "
        f"tools={','.join(route.get('recommended_tools', []))} "
        f"company={preprocessed.get('canonical_company')} "
        f"ticker={preprocessed.get('ticker')}"
    )
    if route.get("needs_clarification"):
        print(f"\nUser: {args.query}\n")
        print("Agent:\n")
        print(route.get("clarification_question"))
        return
    if route.get("intent") == "non_finance":
        print(f"\nUser: {args.query}\n")
        print("Agent:\n")
        print(NON_FINANCE_REPLY)
        return
    stream_agent_response(
        agent,
        query=args.query,
        thread_id=thread_id,
        enable_memory=enable_memory,
        resolved=resolution,
    )


if __name__ == "__main__":
    main()
