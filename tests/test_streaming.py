"""Tests for the streaming provider protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator

from soteria_loop.models import ModelRequest, ModelResponse
from soteria_loop.providers.base import ModelProvider
from soteria_loop.providers.streaming import (
    ModelChunk,
    StreamingModelProvider,
    collect_stream,
)


class _StreamingProvider(StreamingModelProvider, ModelProvider):
    """Provider that yields three chunks then terminates."""

    def __init__(self, chunks: list[ModelChunk]) -> None:
        self._chunks = chunks
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        return ModelResponse(content="fallback")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.calls += 1
        for chunk in self._chunks:
            yield chunk


class _PlainProvider(ModelProvider):
    def __init__(self, response: ModelResponse) -> None:
        self._response = response
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return self._response


def _request() -> ModelRequest:
    return ModelRequest(
        run_id="run-1",
        step=1,
        messages=[{"role": "user", "content": "hi"}],
    )


async def test_collect_stream_assembles_text() -> None:
    provider = _StreamingProvider(
        [
            ModelChunk(text="hello "),
            ModelChunk(text="world"),
            ModelChunk(finish_reason="stop"),
        ]
    )
    response = await collect_stream(provider, _request())
    assert response.content == "hello world"
    assert provider.calls == 1


async def test_collect_stream_falls_back_to_generate() -> None:
    plain = _PlainProvider(ModelResponse(content="from generate"))
    response = await collect_stream(plain, _request())
    assert response.content == "from generate"
    assert plain.calls == 1


async def test_streaming_provider_satisfies_protocol() -> None:
    provider = _StreamingProvider([])
    assert isinstance(provider, StreamingModelProvider)
    assert isinstance(provider, ModelProvider)


async def test_collect_stream_handles_empty_stream() -> None:
    provider = _StreamingProvider([ModelChunk(finish_reason="stop")])
    response = await collect_stream(provider, _request())
    assert response.content == ""
