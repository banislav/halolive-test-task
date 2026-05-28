from deep_agents.config import DeepAgentsSettings, load_env


def test_load_env_loads_api_key_without_overriding_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=from-file",
                "DEEP_AGENTS_MODEL=gpt-test",
                "DEEP_AGENTS_TEMPERATURE=0.25",
            ]
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")

    loaded = load_env(env_file)
    settings = DeepAgentsSettings()

    assert loaded
    assert settings.openai_api_key == "from-process"
    assert settings.model == "gpt-test"
    assert settings.temperature == 0.25


def test_load_env_can_override_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file")
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")

    load_env(env_file, override=True)

    assert DeepAgentsSettings().openai_api_key == "from-file"
