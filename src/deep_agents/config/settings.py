from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepAgentsSettings(BaseSettings):
    """Environment-backed settings for LangChain model construction."""

    model_config = SettingsConfigDict(
        env_prefix="DEEP_AGENTS_",
        env_file=".env",
        extra="ignore",
    )

    model: str = "gpt-4.1-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    provider: str = "openai"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
