from deep_agents.config import DeepAgentsSettings, load_env
from deep_agents.langchain.models import build_chat_model


def test_load_env_loads_api_key_without_overriding_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=from-file",
                "DEEP_AGENTS_MODEL=gpt-test",
                "DEEP_AGENTS_TEMPERATURE=0.25",
            ]
        )
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-process")

    loaded = load_env(env_file)
    settings = DeepAgentsSettings(_env_file=None)

    assert loaded
    assert settings.openrouter_api_key == "from-process"
    assert settings.model == "gpt-test"
    assert settings.temperature == 0.25


def test_load_env_can_override_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-process")

    load_env(env_file, override=True)

    assert DeepAgentsSettings().openrouter_api_key == "from-file"


def test_openrouter_is_the_default_provider_and_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_AGENTS_PROVIDER", raising=False)
    monkeypatch.delenv("DEEP_AGENTS_MODEL", raising=False)
    monkeypatch.delenv("DEEP_AGENTS_TEMPERATURE", raising=False)

    settings = DeepAgentsSettings()

    assert settings.provider == "openrouter"
    assert settings.model == "qwen/qwen3.6-flash"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_openrouter_provider_requires_openrouter_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-that-should-not-be-used")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = DeepAgentsSettings(
        provider="openrouter",
        model="qwen/qwen3.6-flash",
        openrouter_api_key=None,
    )

    try:
        build_chat_model(settings)
    except ValueError as exc:
        assert "OPENROUTER_API_KEY is required" in str(exc)
    else:
        raise AssertionError("expected OpenRouter key validation to fail")
