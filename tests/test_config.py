"""Tests for the HERNNESS_-prefixed environment loader."""

from __future__ import annotations

import pytest

from hernness.config import (
    ConfigError,
    apply_runtime_overrides,
    build_provider_from_env,
    database_path_from_env,
)
from hernness.providers.anthropic import AnthropicProvider
from hernness.providers.minimax import MiniMaxProvider
from hernness.providers.ollama import OllamaProvider
from hernness.providers.openai_compatible import OpenAICompatibleProvider


def test_missing_provider_raises() -> None:
    with pytest.raises(ConfigError, match="HERNNESS_PROVIDER"):
        build_provider_from_env({})


def test_unknown_provider_raises() -> None:
    with pytest.raises(ConfigError, match="HERNNESS_PROVIDER"):
        build_provider_from_env({"HERNNESS_PROVIDER": "azure", "HERNNESS_MODEL": "x"})


def test_missing_model_raises() -> None:
    with pytest.raises(ConfigError, match="HERNNESS_MODEL"):
        build_provider_from_env({"HERNNESS_PROVIDER": "ollama"})


def test_ollama_provider_built_with_no_credentials() -> None:
    provider = build_provider_from_env(
        {"HERNNESS_PROVIDER": "ollama", "HERNNESS_MODEL": "llama3.1"}
    )
    assert isinstance(provider, OllamaProvider)
    assert provider._config.model == "llama3.1"
    assert provider._config.base_url == "http://localhost:11434"


def test_ollama_provider_respects_overrides() -> None:
    provider = build_provider_from_env(
        {
            "HERNNESS_PROVIDER": "ollama",
            "HERNNESS_MODEL": "default-model",
            "HERNNESS_OLLAMA_MODEL": "qwen2.5",
            "HERNNESS_OLLAMA_BASE_URL": "http://gpu.local:11434/",
            "HERNNESS_OLLAMA_API_KEY": "proxy-token",
        }
    )
    assert provider._config.model == "qwen2.5"
    assert provider._config.base_url == "http://gpu.local:11434"
    assert provider._config._api_key == "proxy-token"


def test_minimax_requires_api_key() -> None:
    with pytest.raises(ValueError, match="HERNNESS_MINIMAX_API_KEY"):
        build_provider_from_env({"HERNNESS_PROVIDER": "minimax", "HERNNESS_MODEL": "MiniMax-M3"})


def test_minimax_provider_built() -> None:
    provider = build_provider_from_env(
        {
            "HERNNESS_PROVIDER": "minimax",
            "HERNNESS_MODEL": "MiniMax-M3",
            "HERNNESS_MINIMAX_API_KEY": "test-key",
        }
    )
    assert isinstance(provider, MiniMaxProvider)
    assert provider._config.model == "MiniMax-M3"
    assert provider._config.api_style == "anthropic"
    headers = provider._config.headers()
    assert headers["x-api-key"] == "test-key"


def test_minimax_rejects_unknown_api_style() -> None:
    with pytest.raises(ValueError, match="HERNNESS_MINIMAX_API_STYLE"):
        build_provider_from_env(
            {
                "HERNNESS_PROVIDER": "minimax",
                "HERNNESS_MODEL": "MiniMax-M3",
                "HERNNESS_MINIMAX_API_KEY": "test-key",
                "HERNNESS_MINIMAX_API_STYLE": "bogus",
            }
        )


def test_minimax_openai_style_uses_bearer() -> None:
    provider = build_provider_from_env(
        {
            "HERNNESS_PROVIDER": "minimax",
            "HERNNESS_MODEL": "MiniMax-M3",
            "HERNNESS_MINIMAX_API_KEY": "test-key",
            "HERNNESS_MINIMAX_API_STYLE": "openai",
        }
    )
    assert isinstance(provider, MiniMaxProvider)
    headers = provider._config.headers()
    assert headers["Authorization"] == "Bearer test-key"


def test_anthropic_requires_api_key() -> None:
    with pytest.raises(ValueError, match="HERNNESS_ANTHROPIC_API_KEY"):
        build_provider_from_env(
            {"HERNNESS_PROVIDER": "anthropic", "HERNNESS_MODEL": "claude-sonnet-4-6"}
        )


def test_anthropic_provider_built() -> None:
    provider = build_provider_from_env(
        {
            "HERNNESS_PROVIDER": "anthropic",
            "HERNNESS_MODEL": "claude-sonnet-4-6",
            "HERNNESS_ANTHROPIC_API_KEY": "test-key",
        }
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider._config.model == "claude-sonnet-4-6"
    headers = provider._config.headers()
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"


def test_openai_requires_api_key() -> None:
    with pytest.raises(ValueError, match="HERNNESS_OPENAI_API_KEY"):
        build_provider_from_env({"HERNNESS_PROVIDER": "openai", "HERNNESS_MODEL": "gpt-4o-mini"})


def test_openai_provider_built() -> None:
    provider = build_provider_from_env(
        {
            "HERNNESS_PROVIDER": "openai",
            "HERNNESS_MODEL": "gpt-4o-mini",
            "HERNNESS_OPENAI_API_KEY": "test-key",
            "HERNNESS_OPENAI_BASE_URL": "https://api.example.com/v1/",
        }
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._config.model == "gpt-4o-mini"
    assert provider._config.base_url == "https://api.example.com/v1"
    headers = provider._config.headers()
    assert headers["Authorization"] == "Bearer test-key"


def test_apply_runtime_overrides_parses_integers_and_floats() -> None:
    kwargs: dict[str, object] = {}
    apply_runtime_overrides(
        kwargs,
        {
            "HERNNESS_MAX_TOTAL_TOKENS": "50000",
            "HERNNESS_MAX_RUNTIME_SECONDS": "120.5",
            "HERNNESS_REPEATED_ACTION_LIMIT": "5",
        },
    )
    assert kwargs == {
        "max_total_tokens": 50000,
        "max_runtime_seconds": 120.5,
        "repeated_action_limit": 5,
    }


def test_apply_runtime_overrides_rejects_garbage() -> None:
    with pytest.raises(ConfigError, match="HERNNESS_MAX_TOTAL_TOKENS"):
        apply_runtime_overrides({}, {"HERNNESS_MAX_TOTAL_TOKENS": "not-a-number"})


def test_database_path_from_env_default_none() -> None:
    assert database_path_from_env({}) is None


def test_database_path_from_env_strips_and_returns() -> None:
    assert database_path_from_env({"HERNNESS_DATABASE_PATH": "  /tmp/soteria.db  "}) == (
        "/tmp/soteria.db"
    )
