"""Live integration tests against any OpenAI-compatible chat endpoint.

Auto-skips when ``AVO_OPENAI_API_KEY`` is missing. The base URL
defaults to ``https://api.openai.com/v1``; override with
``AVO_OPENAI_BASE_URL`` to point at a self-hosted vLLM / llama.cpp /
Azure / etc. endpoint.
"""

from __future__ import annotations

import pytest

from avo import ModelResponse
from avo.providers.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)

from .conftest import report_failure, simple_request

pytestmark = pytest.mark.asyncio


def _resolve_key() -> str:
    import os

    value = os.environ.get("AVO_OPENAI_API_KEY", "").strip()
    if not value:
        pytest.skip("AVO_OPENAI_API_KEY not set")
    return value


def _resolve_model() -> str:
    import os

    return os.environ.get("AVO_OPENAI_MODEL", "").strip() or "gpt-4o-mini"


def _resolve_base_url() -> str:
    import os

    return os.environ.get("AVO_OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"


async def test_openai_text_completion() -> None:
    """Plain text completion against the OpenAI-compatible endpoint."""

    config = OpenAICompatibleConfig.model_construct(
        model=_resolve_model(), base_url=_resolve_base_url()
    )
    config._api_key = _resolve_key()
    provider = OpenAICompatibleProvider(config, request_timeout_seconds=60.0)
    try:
        try:
            response = await provider.generate(simple_request())
        except Exception as exc:
            pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()
    assert isinstance(response, ModelResponse)
    assert response.content is not None
    assert response.content.strip() != ""
    if response.usage is not None:
        assert response.usage.input_tokens >= 0
        assert response.usage.output_tokens >= 0


async def test_openai_tool_call_round_trip() -> None:
    """Tool-call round-trip; verifies tool_choice=auto path."""

    from avo import ModelRequest, ToolMetadata

    config = OpenAICompatibleConfig.model_construct(
        model=_resolve_model(), base_url=_resolve_base_url()
    )
    config._api_key = _resolve_key()
    provider = OpenAICompatibleProvider(config, request_timeout_seconds=60.0)
    request = ModelRequest(
        run_id="integration-openai-tool",
        step=1,
        messages=[
            {
                "role": "user",
                "content": ("Use the echo tool with text=ping. Do not write any other text."),
            }
        ],
        tools=[
            ToolMetadata(
                name="echo",
                description="Echo the input back verbatim.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ],
    )
    try:
        try:
            response = await provider.generate(request)
        except Exception as exc:
            pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()
    if response.tool_call is None:
        pytest.skip(f"model did not return tool_call (content={response.content!r})")
    assert response.tool_call.name == "echo"
    assert response.tool_call.arguments == {"text": "ping"}
