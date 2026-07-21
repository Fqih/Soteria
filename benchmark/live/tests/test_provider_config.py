from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_minimax = import_module("examples.live_providers.minimax_provider")
MiniMaxConfig = _minimax.MiniMaxConfig


def test_minimax_openai_config_uses_root_base_url_without_serializing_tokens() -> None:
    config = MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io/",
            "MINIMAX_API_STYLE": "openai",
            "OPENAI_AUTH_TOKEN": "openai-secret",
        }
    )

    assert config.endpoint == "https://api.minimax.io/v1/chat/completions"
    assert config.headers() == {
        "Authorization": "Bearer openai-secret",
        "Content-Type": "application/json",
    }
    serialized = config.model_dump_json()
    assert "openai-secret" not in serialized
    assert "anthropic-secret" not in repr(config)


def test_minimax_anthropic_config_uses_messages_endpoint_and_headers() -> None:
    config = MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io/",
            "MINIMAX_API_STYLE": "anthropic",
            "AUTH_TOKEN": "anthropic-secret",
        }
    )

    assert config.endpoint == "https://api.minimax.io/anthropic/v1/messages"
    assert config.headers() == {
        "x-api-key": "anthropic-secret",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    assert "anthropic-secret" not in config.model_dump_json()
    assert "anthropic-secret" not in repr(config)
    assert "anthropic-secret" not in str(config.model_dump())


def test_minimax_config_defaults_to_openai_style() -> None:
    config = MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io",
            "OPENAI_AUTH_TOKEN": "openai-secret",
        }
    )

    assert config.api_style == "openai"
    assert config.endpoint == "https://api.minimax.io/v1/chat/completions"


def test_minimax_openai_style_allows_absent_anthropic_token() -> None:
    config = MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io",
            "MINIMAX_API_STYLE": "openai",
            "OPENAI_AUTH_TOKEN": "openai-secret",
        }
    )

    assert config.headers() == {
        "Authorization": "Bearer openai-secret",
        "Content-Type": "application/json",
    }


def test_minimax_anthropic_style_allows_absent_openai_token() -> None:
    config = MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io",
            "MINIMAX_API_STYLE": "anthropic",
            "AUTH_TOKEN": "anthropic-secret",
        }
    )

    assert "x-api-key" in config.headers()


def test_minimax_config_requires_selected_openai_token() -> None:
    with pytest.raises(ValueError):
        MiniMaxConfig.from_env(
            {
                "MODEL_MINIMAX": "MiniMax-M3",
                "BASE_URL": "https://api.minimax.io",
                "MINIMAX_API_STYLE": "openai",
                "AUTH_TOKEN": "anthropic-secret",
            }
        )


def test_minimax_config_requires_selected_anthropic_token() -> None:
    with pytest.raises(ValueError):
        MiniMaxConfig.from_env(
            {
                "MODEL_MINIMAX": "MiniMax-M3",
                "BASE_URL": "https://api.minimax.io",
                "MINIMAX_API_STYLE": "anthropic",
                "OPENAI_AUTH_TOKEN": "openai-secret",
            }
        )


def test_minimax_config_rejects_unknown_style() -> None:
    with pytest.raises(ValueError):
        MiniMaxConfig.from_env(
            {
                "MODEL_MINIMAX": "MiniMax-M3",
                "BASE_URL": "https://api.minimax.io",
                "MINIMAX_API_STYLE": "grpc",
                "OPENAI_AUTH_TOKEN": "openai-secret",
            }
        )


def test_minimax_config_requires_model_and_base_url() -> None:
    with pytest.raises(ValueError):
        MiniMaxConfig.from_env(
            {"BASE_URL": "https://api.minimax.io", "OPENAI_AUTH_TOKEN": "openai-secret"}
        )
    with pytest.raises(ValueError):
        MiniMaxConfig.from_env(
            {"MODEL_MINIMAX": "MiniMax-M3", "OPENAI_AUTH_TOKEN": "openai-secret"}
        )
