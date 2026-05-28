from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from deep_agents.config import DeepAgentsSettings, load_env


def build_chat_model(settings: DeepAgentsSettings | None = None) -> BaseChatModel:
    """Build the configured chat model for LangChain agents."""
    load_env()
    resolved = settings or DeepAgentsSettings()
    if resolved.provider != "openai":
        msg = f"unsupported chat model provider: {resolved.provider}"
        raise ValueError(msg)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        msg = "langchain-openai is required to build the OpenAI chat model"
        raise RuntimeError(msg) from exc

    return ChatOpenAI(
        model=resolved.model,
        temperature=resolved.temperature,
        api_key=resolved.openai_api_key,
    )
