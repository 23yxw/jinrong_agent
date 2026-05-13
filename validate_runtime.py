"""One-command runtime validation for all project modules."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


def run_check(name: str, fn):
    try:
        detail = fn()
        print(f"[PASS] {name}: {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {exc}")
        return False


def check_mcp_tools():
    import mcp_tools

    mcp_tools.get_live_finance_researcher_tool()
    return "tool factory is ready"


def check_pdf_pipeline():
    import pdf_pipeline

    converter = pdf_pipeline.build_converter()
    return f"converter={type(converter).__name__}"


def check_vector_store():
    import vector_store

    store = vector_store.build_vector_store()
    return f"vector_store={type(store).__name__}"


def check_retrieval():
    import retrieval

    retrieval.get_deep_rag_search_tool()
    return "tool factory is ready"


def check_agent_app():
    import agent_app

    agent = agent_app.create_default_agent(with_middleware=False)
    return f"agent={type(agent).__name__}"


def main():
    print("===== Runtime Validation =====")
    checks = [
        ("mcp_tools", check_mcp_tools),
        ("pdf_pipeline", check_pdf_pipeline),
        ("vector_store", check_vector_store),
        ("retrieval", check_retrieval),
        ("agent_app", check_agent_app),
    ]
    results = [run_check(name, fn) for name, fn in checks]
    passed = sum(1 for item in results if item)
    print(f"\nSummary: {passed}/{len(results)} passed")


if __name__ == "__main__":
    main()
