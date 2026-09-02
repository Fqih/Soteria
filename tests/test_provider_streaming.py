"""Tests for the HTTP provider streaming layer.

Covers the SSE byte-iterator parser, the per-protocol chunk parsers,
and the end-to-end ``stream()`` coroutine on ``OpenAICompatibleProvider``
and ``AnthropicProvider``. Providers are driven by a fake ``_AsyncHTTPClient``
that returns canned SSE bytes — no real network or httpx client involved.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from avo import ModelRequest
from avo.exceptions import ProviderError
from avo.providers.anthropic import AnthropicConfig, AnthropicProvider
from avo.providers.groq import GroqConfig, GroqProvider
from avo.providers.http_common import (
    _AsyncHTTPClient,
    _StreamContext,
    iter_anthropic_sse_events,
    iter_sse_lines,
    parse_anthropic_stream_event,
    parse_openai_stream_payload,
)
from avo.providers.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)
from avo.providers.streaming import ModelChunk


def _request() -> ModelRequest:
    return ModelRequest(
        run_id="run-1",
        step=1,
        messages=[{"role": "user", "content": "hi"}],
    )


async def _aiter(data: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in data:
        yield chunk


# ---------------------------------------------------------------------------
# Low-level SSE parsers
# ---------------------------------------------------------------------------


async def test_iter_sse_lines_yields_data_only() -> None:
    raw = b'data: {"a":1}\n\n: keep-alive\n\ndata: {"b":2}\n\n'
    lines = [line async for line in iter_sse_lines(_aiter([raw]))]
    assert lines == ['{"a":1}', '{"b":2}']


async def test_iter_sse_lines_handles_split_chunks() -> None:
    raw = _aiter([b'data: {"a', b'":1}\n\ndata: {"b":2', b"}\n\n"])
    lines = [line async for line in iter_sse_lines(raw)]
    assert lines == ['{"a":1}', '{"b":2}']


def test_parse_openai_stream_payload_text_delta() -> None:
    payload = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]}
    )
    chunk = parse_openai_stream_payload(f"data: {payload}")
    assert chunk == ModelChunk(text="hello")


def test_parse_openai_stream_payload_finish_reason() -> None:
    payload = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    chunk = parse_openai_stream_payload(f"data: {payload}")
    assert chunk is not None
    assert chunk.text == ""
    assert chunk.finish_reason == "stop"


def test_parse_openai_stream_payload_done_sentinel() -> None:
    assert parse_openai_stream_payload("data: [DONE]") is None


def test_parse_openai_stream_payload_tool_call_delta() -> None:
    payload = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    chunk = parse_openai_stream_payload(f"data: {payload}")
    assert chunk is not None
    assert chunk.tool_call_delta is not None
    assert chunk.tool_call_delta["id"] == "call_1"
    assert chunk.tool_call_delta["function"]["name"] == "search"


def test_parse_anthropic_text_delta() -> None:
    chunk = parse_anthropic_stream_event(
        "content_block_delta",
        {"delta": {"type": "text_delta", "text": "hel"}},
    )
    assert chunk == ModelChunk(text="hel")


def test_parse_anthropic_tool_use_input_delta() -> None:
    chunk = parse_anthropic_stream_event(
        "content_block_delta",
        {"delta": {"type": "input_json_delta", "partial_json": '{"q"'}},
    )
    assert chunk is not None
    assert chunk.tool_call_delta is not None
    assert chunk.tool_call_delta["arguments_delta"] == '{"q"'


def test_parse_anthropic_message_delta_stop_reason() -> None:
    chunk = parse_anthropic_stream_event(
        "message_delta",
        {"delta": {"stop_reason": "end_turn"}},
    )
    assert chunk is not None
    assert chunk.finish_reason == "end_turn"


def test_parse_anthropic_message_stop() -> None:
    chunk = parse_anthropic_stream_event("message_stop", {})
    assert chunk == ModelChunk(finish_reason="stop")


def test_parse_anthropic_skips_event_types_without_chunks() -> None:
    assert parse_anthropic_stream_event("message_start", {"message": {"id": "x"}}) is None
    assert parse_anthropic_stream_event("ping", {}) is None


async def test_iter_anthropic_sse_events_emits_pairs() -> None:
    raw = (
        b'event: message_start\ndata: {"message":{"id":"m1"}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"delta":{"type":"text_delta","text":"hi"}}\n\n'
        b"event: message_stop\ndata: {}\n\n"
    )
    events = [pair async for pair in iter_anthropic_sse_events(_aiter([raw]))]
    assert [e[0] for e in events] == ["message_start", "content_block_delta", "message_stop"]
    assert events[1][1]["delta"]["text"] == "hi"


# ---------------------------------------------------------------------------
# Fake streaming HTTP client
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text or body.decode("utf-8", errors="replace")
        self.closed = False

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield self._body

    async def aclose(self) -> None:
        self.closed = True


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeStreamingClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.posted: list[dict[str, Any]] = []
        self.streamed: list[dict[str, Any]] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None,  # noqa: ASYNC109 - mirrors the httpx signature
    ) -> _FakeResponse:
        self.posted.append({"url": url, "json": json})
        if not self._responses:
            raise AssertionError("no canned response")
        return self._responses.pop(0)

    def stream(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None,
    ) -> _StreamContext:
        self.streamed.append({"url": url, "json": json})
        if not self._responses:
            raise AssertionError("no canned stream response")
        return _FakeStreamContext(self._responses.pop(0))

    async def aclose(self) -> None:
        self.closed = True


def _openai_sse(*parts: dict[str, Any]) -> bytes:
    blocks = []
    for part in parts:
        blocks.append(f"data: {json.dumps(part)}\n\n")
    blocks.append("data: [DONE]\n\n")
    return "".join(blocks).encode("utf-8")


def _anthropic_sse(*frames: tuple[str, dict[str, Any]]) -> bytes:
    blocks = []
    for event_type, payload in frames:
        blocks.append(f"event: {event_type}\ndata: {json.dumps(payload)}\n\n")
    return "".join(blocks).encode("utf-8")


# ---------------------------------------------------------------------------
# OpenAI-compatible provider streaming
# ---------------------------------------------------------------------------


def _openai_provider(client: _AsyncHTTPClient) -> OpenAICompatibleProvider:
    config = OpenAICompatibleConfig(model="gpt-test")
    config._api_key = "sk-test"
    return OpenAICompatibleProvider(config, max_completion_tokens=32, client=client)


async def test_openai_stream_yields_text_then_finish() -> None:
    body = _openai_sse(
        {"choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    fake = _FakeStreamingClient([_FakeResponse(200, body)])
    provider = _openai_provider(fake)
    chunks = [chunk async for chunk in provider.stream(_request())]
    assert [c.text for c in chunks if c.text] == ["hel", "lo"]
    final = next(c for c in chunks if c.finish_reason)
    assert final.finish_reason == "stop"
    # Payload must request streaming
    assert fake.streamed[0]["json"]["stream"] is True
    await provider.aclose()


async def test_openai_stream_raises_on_non_2xx() -> None:
    fake = _FakeStreamingClient([_FakeResponse(500, b"server error", text="server error")])
    provider = _openai_provider(fake)
    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.stream(_request()):
            pass
    assert "status 500" in str(exc_info.value)
    assert exc_info.value.retryable is True
    await provider.aclose()


async def test_groq_stream_inherits_openai_path() -> None:
    body = _openai_sse(
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    fake = _FakeStreamingClient([_FakeResponse(200, body)])
    config = GroqConfig(model="llama-test")
    config._api_key = "gsk-test"
    provider = GroqProvider(config, max_completion_tokens=16, client=fake)
    chunks = [chunk async for chunk in provider.stream(_request())]
    assert "".join(c.text for c in chunks) == "ok"
    assert fake.streamed[0]["json"]["stream"] is True
    await provider.aclose()


# ---------------------------------------------------------------------------
# Anthropic provider streaming
# ---------------------------------------------------------------------------


def _anthropic_provider(client: _AsyncHTTPClient) -> AnthropicProvider:
    config = AnthropicConfig(model="claude-test")
    config._api_key = "sk-ant"
    return AnthropicProvider(config, max_completion_tokens=32, client=client)


async def test_anthropic_stream_yields_text_then_finish() -> None:
    body = _anthropic_sse(
        ("message_start", {"message": {"id": "m_1"}}),
        (
            "content_block_delta",
            {"delta": {"type": "text_delta", "text": "hel"}},
        ),
        (
            "content_block_delta",
            {"delta": {"type": "text_delta", "text": "lo"}},
        ),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}}),
        ("message_stop", {}),
    )
    fake = _FakeStreamingClient([_FakeResponse(200, body)])
    provider = _anthropic_provider(fake)
    chunks = [chunk async for chunk in provider.stream(_request())]
    assert [c.text for c in chunks if c.text] == ["hel", "lo"]
    finish_reasons = [c.finish_reason for c in chunks if c.finish_reason]
    assert finish_reasons == ["end_turn", "stop"]
    assert fake.streamed[0]["json"]["stream"] is True
    await provider.aclose()


async def test_anthropic_stream_emits_tool_input_delta() -> None:
    body = _anthropic_sse(
        (
            "content_block_delta",
            {"delta": {"type": "input_json_delta", "partial_json": '{"q":"hi"}'}},
        ),
        ("message_stop", {}),
    )
    fake = _FakeStreamingClient([_FakeResponse(200, body)])
    provider = _anthropic_provider(fake)
    chunks = [chunk async for chunk in provider.stream(_request())]
    tool_chunks = [c for c in chunks if c.tool_call_delta is not None]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_call_delta is not None
    assert tool_chunks[0].tool_call_delta["arguments_delta"] == '{"q":"hi"}'
    await provider.aclose()


async def test_anthropic_stream_raises_on_non_2xx() -> None:
    fake = _FakeStreamingClient([_FakeResponse(429, b"rate limited", text="rate limited")])
    provider = _anthropic_provider(fake)
    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.stream(_request()):
            pass
    assert "status 429" in str(exc_info.value)
    assert exc_info.value.retryable is True
    await provider.aclose()


# ---------------------------------------------------------------------------
# collect_stream() end-to-end with streaming providers
# ---------------------------------------------------------------------------


async def test_collect_stream_consumes_provider_stream() -> None:
    from avo.providers.streaming import collect_stream

    body = _openai_sse(
        {"choices": [{"index": 0, "delta": {"content": "abc"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "def"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    fake = _FakeStreamingClient([_FakeResponse(200, body)])
    provider = _openai_provider(fake)
    response = await collect_stream(provider, _request())
    assert response.content == "abcdef"
    await provider.aclose()
