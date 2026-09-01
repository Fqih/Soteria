"""Live integration tests against the Anthropic Messages API.

Auto-skips when ``AVO_ANTHROPIC_API_KEY`` is missing.
"""

from __future__ import annotations

import pytest

from avo import ModelResponse
from avo.providers.anthropic import AnthropicConfig, AnthropicProvider

from .conftest import report_failure, simple_request

pytestmark = pytest.mark.asyncio


def _resolve_key() -> str:
    import os

    value = os.environ.get("AVO_ANTHROPIC_API_KEY", "").strip()
    if not value:
        pytest.skip("AVO_ANTHROPIC_API_KEY not set")
    return value


def _resolve_model() -> str:
    import os

    return os.environ.get("AVO_ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6"


async def test_anthropic_text_completion() -> None:
    """Plain text completion; verifies auth and model availability."""

    config = AnthropicConfig.model_construct(model=_resolve_model())
    config._api_key = _resolve_key()
    provider = AnthropicProvider(config, request_timeout_seconds=60.0)
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


async def test_anthropic_tool_use_block() -> None:
    """Tool-use round-trip; verifies tool schema and content-block parsing."""

    from avo import ModelRequest, ToolMetadata

    config = AnthropicConfig.model_construct(model=_resolve_model())
    config._api_key = _resolve_key()
    provider = AnthropicProvider(config, request_timeout_seconds=60.0)
    request = ModelRequest(
        run_id="integration-anthropic-tool",
        step=1,
        messages=[
            {
                "role": "user",
                "content": ("Use the echo tool: pass text=ping. Do not write any other text."),
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
        pytest.skip(f"model did not return tool_use (content={response.content!r})")
    assert response.tool_call.name == "echo"
    assert response.tool_call.arguments == {"text": "ping"}
