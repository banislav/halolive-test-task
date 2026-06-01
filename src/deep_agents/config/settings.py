from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepAgentsSettings(BaseSettings):
    """Environment-backed settings for LangChain model construction."""

    model_config = SettingsConfigDict(
        env_prefix="DEEP_AGENTS_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    model: str = "qwen/qwen3.6-flash"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    provider: str = "openrouter"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DEEP_AGENTS_OPENAI_API_KEY"),
    )
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "DEEP_AGENTS_OPENROUTER_API_KEY"),
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "halolive-deep-agents"
