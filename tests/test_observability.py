"""Tests for ``avo.observability`` — OpenTelemetry integration."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from avo.observability import (
    OtelDisabledError,
    configure_tracer,
    is_enabled,
    record_tool_call,
    record_usage,
    span_for_turn,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AVO_OTEL_ENABLED", raising=False)


def test_is_enabled_false_when_env_unset() -> None:
    assert is_enabled() is False
    configure_tracer()
    assert is_enabled() is False


def test_configure_tracer_returns_noop_when_disabled() -> None:
    tracer = configure_tracer()
    assert tracer is not None
    with span_for_turn("run-1") as span:
        span.set_attribute("k", "v")
    assert True  # did not raise


def test_span_for_turn_accepts_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVO_OTEL_ENABLED", "")
    configure_tracer()
    # Even when disabled the helper must accept the documented kwargs.
    with span_for_turn("run-2", provider="ollama", model="llama3") as span:
        record_tool_call(span, "read_file", call_id="abc")
        record_usage(span, input_tokens=10, output_tokens=5, cost_usd=0.001)
    assert True


def test_record_usage_ignores_none() -> None:
    span = span_for_turn("run-3")
    span.__enter__()
    try:
        record_usage(span)
        record_usage(span, input_tokens=None, output_tokens=None, cost_usd=None)
    finally:
        span.__exit__(None, None, None)
    assert True


def test_configure_tracer_raises_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AVO_OTEL_ENABLED=1`` without the SDK raises a clear error."""

    monkeypatch.setenv("AVO_OTEL_ENABLED", "1")
    with patch.dict(os.environ, {}, clear=False):
        # Force ImportError on the SDK import.
        import builtins

        original_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("opentelemetry"):
                raise ImportError("simulated missing sdk")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=fake_import),
            pytest.raises(OtelDisabledError),
        ):
            configure_tracer()


def test_observability_does_not_break_runtime() -> None:
    """Sanity check: the noop path is a true no-op for the agent loop."""

    from avo.models import ModelResponse
    from avo.providers.base import ModelProvider
    from avo.runtime import AgentRuntime

    class Stub(ModelProvider):
        async def generate(self, request: object) -> ModelResponse:  # type: ignore[override]
            return ModelResponse(content="ok")

    async def go() -> str:
        rt = AgentRuntime(provider=Stub())
        result = await rt.run("hello")
        return result.status.value

    assert asyncio.run(go()) == "completed"
