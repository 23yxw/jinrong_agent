"""Model and embedding factory with environment-variable based provider switch."""

from __future__ import annotations

import os

from config import GEMINI_CHAT_MODEL, GEMINI_EMBED_MODEL
from runtime_checks import assert_runtime_ready


def _chat_provider() -> str:
    return os.getenv("MODEL_PROVIDER", "google").strip().lower()


def _embedding_provider() -> str:
    value = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    return value or _chat_provider()


def get_chat_model(temperature: float = 0):
    provider = _chat_provider()
    if provider == "google":
        assert_runtime_ready(
            stage="model_factory.get_chat_model.google",
            packages=["langchain_google_genai"],
            env_vars=["GOOGLE_API_KEY"],
        )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = (
            os.getenv("GOOGLE_CHAT_MODEL", "").strip()
            or os.getenv("CHAT_MODEL", "").strip()
            or GEMINI_CHAT_MODEL
        )
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if provider == "openai":
        assert_runtime_ready(
            stage="model_factory.get_chat_model.openai",
            packages=["langchain_openai"],
            env_vars=["OPENAI_API_KEY"],
        )
        from langchain_openai import ChatOpenAI

        model_name = (
            os.getenv("OPENAI_CHAT_MODEL", "").strip()
            or os.getenv("CHAT_MODEL", "").strip()
            or "gpt-4o-mini"
        )
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    if provider in {"qwen", "dashscope", "bailian"}:
        assert_runtime_ready(
            stage="model_factory.get_chat_model.qwen",
            packages=["langchain_openai"],
            env_vars=["DASHSCOPE_API_KEY"],
        )
        from langchain_openai import ChatOpenAI

        model_name = (
            os.getenv("QWEN_CHAT_MODEL", "").strip()
            or os.getenv("CHAT_MODEL", "").strip()
            or "qwen3-omni-flash"
        )
        base_url = (
            os.getenv("QWEN_BASE_URL", "").strip()
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )

    raise RuntimeError(
        "Unsupported MODEL_PROVIDER: "
        f"{provider}. Supported values: google, openai, qwen."
    )


def get_embedding_model():
    provider = _embedding_provider()
    if provider == "google":
        assert_runtime_ready(
            stage="model_factory.get_embedding_model.google",
            packages=["langchain_google_genai"],
            env_vars=["GOOGLE_API_KEY"],
        )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model_name = (
            os.getenv("GOOGLE_EMBED_MODEL", "").strip()
            or os.getenv("EMBED_MODEL", "").strip()
            or GEMINI_EMBED_MODEL
        )
        return GoogleGenerativeAIEmbeddings(model=model_name)

    if provider == "openai":
        assert_runtime_ready(
            stage="model_factory.get_embedding_model.openai",
            packages=["langchain_openai"],
            env_vars=["OPENAI_API_KEY"],
        )
        from langchain_openai import OpenAIEmbeddings

        model_name = (
            os.getenv("OPENAI_EMBED_MODEL", "").strip()
            or os.getenv("EMBED_MODEL", "").strip()
            or "text-embedding-3-large"
        )
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        return OpenAIEmbeddings(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    if provider in {"qwen", "dashscope", "bailian"}:
        assert_runtime_ready(
            stage="model_factory.get_embedding_model.qwen",
            packages=["langchain_openai"],
            env_vars=["DASHSCOPE_API_KEY"],
        )
        from langchain_openai import OpenAIEmbeddings

        model_name = (
            os.getenv("QWEN_EMBED_MODEL", "").strip()
            or os.getenv("EMBED_MODEL", "").strip()
            or "text-embedding-v3"
        )
        base_url = (
            os.getenv("QWEN_BASE_URL", "").strip()
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        return OpenAIEmbeddings(
            model=model_name,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )

    raise RuntimeError(
        "Unsupported EMBEDDING_PROVIDER: "
        f"{provider}. Supported values: google, openai, qwen."
    )
