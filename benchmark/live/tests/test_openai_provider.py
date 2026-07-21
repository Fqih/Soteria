from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from soteria import ModelRequest, TokenUsage, ToolCall, ToolMetadata
from soteria.exceptions import ProviderError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_openai = import_module("examples.live_providers.openai_provider")
OpenAIConfig = _openai.OpenAIConfig
OpenAIProvider = _openai.OpenAIProvider


class _FakeResponse:
    """Minimal response object returned by the offline fake client."""

    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("response body is not valid JSON")
        return self._payload


class _FakeClient:
    """Record requests and return scripted responses without network access."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[SimpleNamespace] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Any = None,  # noqa: ASYNC109 - mirrors the httpx client signature
    ) -> _FakeResponse:
        self.requests.append(SimpleNamespace(url=url, headers=headers, json=json, timeout=timeout))
        return next(self._responses)

    async def aclose(self) -> None:
        self.closed = True


def _config() -> Any:
    return OpenAIConfig.from_env(
        {
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "openai-secret",
        }
    )


def _request() -> ModelRequest:
    return ModelRequest(
        run_id="run-1",
        step=1,
        messages=[{"role": "user", "content": "add two and three"}],
        tools=[
            ToolMetadata(
                name="add",
                description="Add two integers.",
                input_schema={"type": "object"},
            )
        ],
    )


def test_openai_config_uses_real_openai_defaults_and_hides_key() -> None:
    config = OpenAIConfig.from_env(
        {
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": "openai-secret",
        }
    )

    assert config.endpoint == "https://api.openai.com/v1/chat/completions"
    assert config.headers() == {
        "Authorization": "Bearer openai-secret",
        "Content-Type": "application/json",
    }
    assert "openai-secret" not in config.model_dump_json()
    assert "openai-secret" not in repr(config)


def test_openai_config_normalizes_custom_base_url() -> None:
    config = OpenAIConfig.from_env(
        {
            "OPENAI_MODEL": "custom-model",
            "OPENAI_API_KEY": "openai-secret",
            "OPENAI_BASE_URL": "https://openai.example/v1/",
        }
    )

    assert config.base_url == "https://openai.example/v1"
    assert config.endpoint == "https://openai.example/v1/chat/completions"


async def test_openai_provider_posts_to_default_endpoint_and_parses_final_response() -> None:
    client = _FakeClient(
        [
            _FakeResponse(
                200,
                {
                    "choices": [{"message": {"role": "assistant", "content": "finished"}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                },
            )
        ]
    )
    provider = OpenAIProvider(
        _config(),
        max_completion_tokens=256,
        request_timeout_seconds=12.5,
        client=client,
    )

    response = await provider.generate(_request())

    assert response.content == "finished"
    assert response.usage == TokenUsage(input_tokens=11, output_tokens=7)
    sent = client.requests[0]
    assert sent.url == "https://api.openai.com/v1/chat/completions"
    assert sent.headers == {
        "Authorization": "Bearer openai-secret",
        "Content-Type": "application/json",
    }
    assert sent.json["model"] == "gpt-4o-mini"
    assert sent.json["max_completion_tokens"] == 256
    assert sent.json["messages"] == [{"role": "user", "content": "add two and three"}]
    assert sent.json["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert sent.timeout == 12.5


async def test_openai_provider_parses_tool_call_response() -> None:
    client = _FakeClient(
        [
            _FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"left": 2, "right": 3}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        ]
    )
    provider = OpenAIProvider(_config(), client=client)

    response = await provider.generate(_request())

    assert response.content is None
    assert response.tool_call == ToolCall(
        tool_call_id="call-1",
        name="add",
        arguments={"left": 2, "right": 3},
    )


async def test_openai_provider_redacts_credentials_from_http_errors() -> None:
    client = _FakeClient(
        [
            _FakeResponse(
                401,
                text="rejected openai-secret and sk-abcdefgh1234",
            )
        ]
    )
    provider = OpenAIProvider(_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.retryable is False
    assert "openai-secret" not in str(caught.value)
    assert "sk-abcdefgh1234" not in str(caught.value)


async def test_openai_provider_does_not_close_injected_client() -> None:
    client = _FakeClient([])
    provider = OpenAIProvider(_config(), client=client)

    await provider.aclose()

    assert client.closed is False
