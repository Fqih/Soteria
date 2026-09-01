"""Live integration tests against a local Ollama daemon.

Auto-skips when ``AVO_OLLAMA_BASE_URL`` is unreachable or the
configured model has not been pulled. No API key required.
"""

from __future__ import annotations

import pytest

from avo import ModelResponse

from .conftest import (
    ollama_has_model,
    ollama_model,
    ollama_reachable,
    ollama_url,
    report_failure,
    simple_request,
)

pytestmark = pytest.mark.asyncio


async def test_ollama_text_completion() -> None:
    """Issue a simple text request; expect a non-empty content string."""

    if not ollama_reachable(ollama_url()):
        pytest.skip(f"Ollama daemon unreachable at {ollama_url()}")
    if not ollama_has_model(ollama_url(), ollama_model()):
        pytest.skip(f"Ollama model {ollama_model()!r} not pulled locally")

    from avo.providers.ollama import OllamaConfig, OllamaProvider

    config = OllamaConfig(model=ollama_model(), base_url=ollama_url())
    provider = OllamaProvider(config, request_timeout_seconds=60.0)

    try:
        response = await provider.generate(simple_request())
    except Exception as exc:  # ProviderError or transport
        pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()

    assert isinstance(response, ModelResponse)
    assert response.content is not None
    assert response.content.strip() != ""
    # Ollama usage accounting is best-effort; presence is what matters.
    if response.usage is not None:
        assert response.usage.input_tokens >= 0
        assert response.usage.output_tokens >= 0


async def test_ollama_tool_call_returns_typed_arguments() -> None:
    """Round-trip a tool call: model picks the tool, arguments parse as dict.

    Uses a prompt that nudges the model toward the registered tool. If
    the local model is too small or misaligned, the test reports the
    actual content so the operator can adjust the prompt.
    """

    if not ollama_reachable(ollama_url()):
        pytest.skip(f"Ollama daemon unreachable at {ollama_url()}")
    if not ollama_has_model(ollama_url(), ollama_model()):
        pytest.skip(f"Ollama model {ollama_model()!r} not pulled locally")

    from avo import ModelRequest, ToolMetadata
    from avo.providers.ollama import OllamaConfig, OllamaProvider

    request = ModelRequest(
        run_id="integration-tool-run",
        step=1,
        messages=[
            {
                "role": "user",
                "content": "Call the echo tool with text=ping and nothing else.",
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
    config = OllamaConfig(model=ollama_model(), base_url=ollama_url())
    provider = OllamaProvider(config, request_timeout_seconds=60.0)

    try:
        response = await provider.generate(request)
    except Exception as exc:
        pytest.fail(report_failure(exc))  # type: ignore[arg-type]
    finally:
        await provider.aclose()

    if response.tool_call is None:
        pytest.skip(
            f"model did not return a tool call (content={response.content!r}); "
            "smaller models often refuse tool-use prompts"
        )
    assert response.tool_call.name == "echo"
    assert response.tool_call.arguments == {"text": "ping"}
