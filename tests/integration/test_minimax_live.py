"""Live integration tests against the MiniMax API.

Covers both the **OpenAI-compatible** endpoint (``/v1/chat/completions``)
and the **Anthropic-compatible** endpoint (``/anthropic/v1/messages``).
Auto-skips when ``AVO_MINIMAX_API_KEY`` is missing. On a non-2xx
response the test prints the unredacted provider error so the operator
can diagnose 400 / 401 / 429 problems without re-running with extra
logging.
"""

from __future__ import annotations

import pytest

from avo import ModelResponse
from avo.providers.minimax import MiniMaxConfig, MiniMaxProvider

from .conftest import report_failure, simple_request

pytestmark = pytest.mark.asyncio


def _resolve_key() -> str:
    import os

    value = os.environ.get("AVO_MINIMAX_API_KEY", "").strip()
    if not value:
        pytest.skip("AVO_MINIMAX_API_KEY not set")
    return value


def _resolve_model() -> str:
    import os

    return os.environ.get("AVO_MINIMAX_MODEL", "").strip() or "MiniMax-M3"


def _resolve_base_url() -> str:
    import os

    return os.environ.get("AVO_MINIMAX_BASE_URL", "").strip() or "https://api.minimax.io"


def _make_provider(style: str, *, key: str, model: str, base_url: str) -> MiniMaxProvider:
    config = MiniMaxConfig.model_construct(model=model, base_url=base_url, api_style=style)  # type: ignore[arg-type]
    config._avo_api_key = key
    return MiniMaxProvider(config, request_timeout_seconds=60.0)


async def test_minimax_anthropic_text() -> None:
    """Plain text completion against the Anthropic-compatible endpoint."""

    provider = _make_provider(
        "anthropic",
        key=_resolve_key(),
        model=_resolve_model(),
        base_url=_resolve_base_url(),
    )
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


async def test_minimax_anthropic_tool_use() -> None:
    """Tool-use round-trip against the Anthropic-compatible endpoint."""

    provider = _make_provider(
        "anthropic",
        key=_resolve_key(),
        model=_resolve_model(),
        base_url=_resolve_base_url(),
    )
    try:
        try:
            response = await provider.generate(
                _request_with_strong_tool_prompt(),
            )
        except Exception as exc:
            pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()

    if response.tool_call is None:
        pytest.skip(f"model did not return a tool_use block (content={response.content!r})")
    assert response.tool_call.name == "echo"
    assert response.tool_call.arguments == {"text": "ping"}


async def test_minimax_openai_text() -> None:
    """Plain text completion against the OpenAI-compatible endpoint.

    Switch ``AVO_MINIMAX_API_STYLE=openai`` in the env to drive this
    path; otherwise it still runs because we construct the provider
    directly here.
    """

    provider = _make_provider(
        "openai",
        key=_resolve_key(),
        model=_resolve_model(),
        base_url=_resolve_base_url(),
    )
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


async def test_minimax_openai_tool_call() -> None:
    """Tool-call round-trip against the OpenAI-compatible endpoint."""

    provider = _make_provider(
        "openai",
        key=_resolve_key(),
        model=_resolve_model(),
        base_url=_resolve_base_url(),
    )
    try:
        try:
            response = await provider.generate(
                _request_with_strong_tool_prompt(),
            )
        except Exception as exc:
            pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()

    if response.tool_call is None:
        pytest.skip(f"model did not return a tool_call (content={response.content!r})")
    assert response.tool_call.name == "echo"
    assert response.tool_call.arguments == {"text": "ping"}


async def test_minimax_anthropic_does_not_400_on_minimum_payload() -> None:
    """Smoke test for the Anthropic-style minimum-request payload.

    The MiniMax Anthropic-compatible endpoint has historically rejected
    requests with 400 for one of these reasons:

    - missing ``system`` field (some implementations require it explicitly)
    - ``max_tokens`` smaller than the model's lower bound
    - tool ``input_schema`` missing ``type: object``

    This test sends the minimum legal Anthropic payload (no tools, single
    user message, ``max_tokens=64``) so a regression is easy to bisect.
    """

    from avo import ModelRequest

    provider = _make_provider(
        "anthropic",
        key=_resolve_key(),
        model=_resolve_model(),
        base_url=_resolve_base_url(),
    )
    request = ModelRequest(
        run_id="integration-min-anthropic",
        step=1,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        tools=[],
    )
    try:
        try:
            response = await provider.generate(request)
        except Exception as exc:
            pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()
    assert response.content is not None
    assert "pong" in response.content.lower()


def _request_with_strong_tool_prompt():
    """Build a request whose prompt strongly nudges the tool-use path."""

    from avo import ModelRequest, ToolMetadata

    return ModelRequest(
        run_id="integration-tool-minimax",
        step=1,
        messages=[
            {
                "role": "user",
                "content": (
                    "You must call the echo tool now with text=ping. "
                    "Do not write any other text before or after."
                ),
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
