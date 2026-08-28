"""Streaming protocol for providers.

A streaming provider yields ``ModelChunk`` events instead of (or in
addition to) returning a final :class:`ModelResponse`. The runtime
falls back to ``generate`` when a provider only implements the
non-streaming API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from soteria_loop.models import ModelRequest, ModelResponse

from .base import ModelProvider


@dataclass(frozen=True)
class ModelChunk:
    """One chunk of streamed output.

    ``text`` is non-empty for incremental text deltas. ``finish_reason``
    is set on the terminal chunk. ``tool_call_delta`` carries partial
    tool-call JSON when the provider streams tool arguments.
    """

    text: str = ""
    finish_reason: str | None = None
    tool_call_delta: dict[str, JsonValue] | None = None


@runtime_checkable
class StreamingModelProvider(ModelProvider, Protocol):
    """A provider that can stream output via :meth:`stream`.

    The method is declared as a plain (non-async) callable that returns
    an :class:`AsyncIterator` so implementations may use ``async def``
    with ``yield`` (the canonical Python form) without tripping mypy.
    """

    def stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelChunk]:  # pragma: no cover - protocol
        ...


async def collect_stream(
    provider: ModelProvider,
    request: ModelRequest,
) -> ModelResponse:
    """Run ``provider.stream`` (if available) and assemble a final response.

    Providers that do not implement :class:`StreamingModelProvider` fall
    back to :meth:`ModelProvider.generate`. The returned
    :class:`ModelResponse` carries the concatenated text plus the
    finish reason from the terminal chunk.
    """

    if isinstance(provider, StreamingModelProvider):
        text_parts: list[str] = []
        async for chunk in provider.stream(request):
            if chunk.text:
                text_parts.append(chunk.text)
        return ModelResponse(content="".join(text_parts))
    return await provider.generate(request)


__all__ = ["ModelChunk", "StreamingModelProvider", "collect_stream"]
