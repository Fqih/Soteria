from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from hernness import ModelRequest, TokenUsage, ToolCall, ToolMetadata
from hernness.exceptions import ProviderError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_minimax = import_module("examples.live_providers.minimax_provider")
MiniMaxConfig = _minimax.MiniMaxConfig
MiniMaxProvider = _minimax.MiniMaxProvider


class _FakeResponse:
    """Minimal stand-in for an httpx response used offline."""

    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("response body is not valid JSON")
        return self._payload


class _FakeClient:
    """Records the single request it receives and returns a scripted response."""

    def __init__(
        self, response: _FakeResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
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
        self.requests.append(SimpleNamespace(url=url, headers=headers, json=json))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    async def aclose(self) -> None:
        self.closed = True


def _openai_config() -> Any:
    return MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io/",
            "MINIMAX_API_STYLE": "openai",
            "OPENAI_AUTH_TOKEN": "openai-secret",
        }
    )


def _anthropic_config() -> Any:
    return MiniMaxConfig.from_env(
        {
            "MODEL_MINIMAX": "MiniMax-M3",
            "BASE_URL": "https://api.minimax.io/",
            "MINIMAX_API_STYLE": "anthropic",
            "AUTH_TOKEN": "anthropic-secret",
        }
    )


def _text_request() -> ModelRequest:
    return ModelRequest(
        run_id="run-1",
        step=1,
        messages=[{"role": "user", "content": "hi"}],
    )


def _tool_history_request() -> ModelRequest:
    return ModelRequest(
        run_id="run-1",
        step=2,
        messages=[
            {"role": "user", "content": "add"},
            {
                "role": "assistant",
                "tool_call": {
                    "tool_call_id": "call-1",
                    "name": "add",
                    "arguments": {"left": 2, "right": 3},
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "add",
                "content": {"sum": 5},
            },
        ],
        tools=[
            ToolMetadata(
                name="add",
                description="Add two integers.",
                input_schema={"type": "object"},
            )
        ],
    )


async def test_openai_provider_posts_expected_request() -> None:
    client = _FakeClient(
        _FakeResponse(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )
    )
    provider = MiniMaxProvider(_openai_config(), client=client)

    response = await provider.generate(_text_request())

    assert response.content == "done"
    assert response.usage == TokenUsage(input_tokens=5, output_tokens=3)
    request = client.requests[0]
    assert request.url == "https://api.minimax.io/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer openai-secret"
    assert request.json["model"] == "MiniMax-M3"
    assert request.json["max_completion_tokens"] == 1024


async def test_anthropic_provider_posts_expected_request_and_blocks() -> None:
    client = _FakeClient(
        _FakeResponse(
            200,
            {
                "content": [{"type": "text", "text": "final"}],
                "usage": {"input_tokens": 4, "output_tokens": 6},
            },
        )
    )
    provider = MiniMaxProvider(_anthropic_config(), max_completion_tokens=256, client=client)

    response = await provider.generate(_tool_history_request())

    assert response.content == "final"
    assert response.usage == TokenUsage(input_tokens=4, output_tokens=6)
    request = client.requests[0]
    assert request.url == "https://api.minimax.io/anthropic/v1/messages"
    assert request.headers["x-api-key"] == "anthropic-secret"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.json["model"] == "MiniMax-M3"
    assert request.json["max_tokens"] == 256
    assert request.json["messages"] == [
        {"role": "user", "content": "add"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "add",
                    "input": {"left": 2, "right": 3},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": '{"sum": 5}',
                }
            ],
        },
    ]
    assert request.json["tools"] == [
        {
            "name": "add",
            "description": "Add two integers.",
            "input_schema": {"type": "object"},
        }
    ]


async def test_anthropic_provider_parses_first_tool_use_block() -> None:
    client = _FakeClient(
        _FakeResponse(
            200,
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-9",
                        "name": "add",
                        "input": {"left": 2, "right": 3},
                    },
                    {
                        "type": "tool_use",
                        "id": "call-10",
                        "name": "ignored",
                        "input": {},
                    },
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )
    )
    provider = MiniMaxProvider(_anthropic_config(), client=client)

    response = await provider.generate(_text_request())

    assert response.content is None
    assert response.tool_call == ToolCall(
        tool_call_id="call-9",
        name="add",
        arguments={"left": 2, "right": 3},
    )


async def test_provider_raises_retryable_error_on_rate_limit() -> None:
    client = _FakeClient(_FakeResponse(429, text="rate limited"))
    provider = MiniMaxProvider(_openai_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_text_request())

    assert caught.value.retryable is True


async def test_provider_raises_retryable_error_on_server_error() -> None:
    client = _FakeClient(_FakeResponse(503, text="unavailable"))
    provider = MiniMaxProvider(_openai_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_text_request())

    assert caught.value.retryable is True


async def test_provider_raises_non_retryable_error_on_client_error() -> None:
    client = _FakeClient(_FakeResponse(400, text="bad request"))
    provider = MiniMaxProvider(_openai_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_text_request())

    assert caught.value.retryable is False


async def test_provider_wraps_transport_error_as_retryable() -> None:
    client = _FakeClient(error=httpx.ConnectError("connection refused"))
    provider = MiniMaxProvider(_openai_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_text_request())

    assert caught.value.retryable is True


async def test_provider_wraps_invalid_json_as_non_retryable() -> None:
    client = _FakeClient(_FakeResponse(200, payload=None, text="not json"))
    provider = MiniMaxProvider(_openai_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_text_request())

    assert caught.value.retryable is False


async def test_provider_error_never_exposes_authorization_header() -> None:
    leaking_body = "rejected token Authorization: Bearer openai-secret"
    client = _FakeClient(_FakeResponse(401, text=leaking_body))
    provider = MiniMaxProvider(_openai_config(), client=client)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(_text_request())

    assert "openai-secret" not in str(caught.value)


async def test_provider_does_not_close_injected_client() -> None:
    client = _FakeClient(_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}))
    provider = MiniMaxProvider(_openai_config(), client=client)

    await provider.aclose()

    assert client.closed is False


async def test_provider_closes_owned_client() -> None:
    provider = MiniMaxProvider(_openai_config())

    await provider.aclose()  # should create and close an httpx client without network use
