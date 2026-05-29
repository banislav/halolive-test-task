from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from deep_agents.config import DeepAgentsSettings, load_env
from deep_agents.instrumentation import get_logger

logger = get_logger(__name__)


def build_chat_model(settings: DeepAgentsSettings | None = None) -> BaseChatModel:
    """Build the configured chat model for LangChain agents."""
    load_env()
    resolved = settings or DeepAgentsSettings()
    logger.info(
        "building chat model",
        extra={"provider": resolved.provider, "model": resolved.model},
    )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        msg = "langchain-openai is required to build the OpenAI chat model"
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if resolved.provider == "openrouter":
        if not resolved.openrouter_api_key:
            msg = "OPENROUTER_API_KEY is required when DEEP_AGENTS_PROVIDER=openrouter"
            logger.error(msg, extra={"provider": resolved.provider, "model": resolved.model})
            raise ValueError(msg)
        headers = {"X-Title": resolved.openrouter_app_name}
        if resolved.openrouter_site_url:
            headers["HTTP-Referer"] = resolved.openrouter_site_url
        return ChatOpenAI(
            model=resolved.model,
            temperature=resolved.temperature,
            api_key=resolved.openrouter_api_key,
            base_url=resolved.openrouter_base_url,
            default_headers=headers,
        )

    if resolved.provider != "openai":
        msg = f"unsupported chat model provider: {resolved.provider}"
        logger.error(msg, extra={"provider": resolved.provider, "model": resolved.model})
        raise ValueError(msg)

    return ChatOpenAI(
        model=resolved.model,
        temperature=resolved.temperature,
        api_key=resolved.openai_api_key,
    )
