from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

from hernness import ModelRequest, TokenUsage, ToolCall, ToolMetadata
from hernness.exceptions import ProviderError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_common = import_module("examples.live_providers.common")
build_openai_payload = _common.build_openai_payload
json_safe_content = _common.json_safe_content
parse_openai_response = _common.parse_openai_response
redact_text = _common.redact_text


def test_build_openai_payload_converts_tool_history() -> None:
    request = ModelRequest(
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

    payload = build_openai_payload("fixture-model", request, 128)

    assert payload["model"] == "fixture-model"
    assert payload["max_completion_tokens"] == 128
    assert payload["stream"] is False
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "user", "content": "add"}
    assert messages[1] == {
        "role": "assistant",
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
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"sum": 5}',
    }
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert payload["tool_choice"] == "auto"


def test_build_openai_payload_preserves_text_history_without_tools() -> None:
    request = ModelRequest(
        run_id="run-1",
        step=1,
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )

    payload = build_openai_payload("fixture-model", request, 64)

    assert payload["messages"] == request.messages
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_json_safe_content_serializes_non_text_values() -> None:
    assert json_safe_content({"sum": 5}) == '{"sum": 5}'
    assert json_safe_content("plain text") == "plain text"


def test_parse_openai_response_parses_final_text() -> None:
    response = parse_openai_response(
        {"choices": [{"message": {"role": "assistant", "content": "finished"}}]}
    )

    assert response.content == "finished"
    assert response.tool_call is None


def test_parse_openai_response_parses_first_tool_call() -> None:
    response = parse_openai_response(
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
                            },
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {"name": "ignored", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ]
        }
    )

    assert response.content is None
    assert response.tool_call == ToolCall(
        tool_call_id="call-1",
        name="add",
        arguments={"left": 2, "right": 3},
    )


def test_parse_openai_response_converts_usage() -> None:
    response = parse_openai_response(
        {
            "choices": [{"message": {"content": "finished"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )

    assert response.usage == TokenUsage(input_tokens=11, output_tokens=7)


def test_parse_openai_response_leaves_missing_usage_unavailable() -> None:
    response = parse_openai_response({"choices": [{"message": {"content": "finished"}}]})

    assert response.usage is None


def test_parse_openai_response_rejects_empty_choices() -> None:
    with pytest.raises(ProviderError) as caught:
        parse_openai_response({"choices": []})

    assert caught.value.retryable is False


def test_parse_openai_response_rejects_malformed_tool_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "add", "arguments": "not-json"},
                        }
                    ]
                }
            }
        ]
    }

    with pytest.raises(ProviderError) as caught:
        parse_openai_response(payload)

    assert caught.value.retryable is False


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("Authorization: Bearer live-secret-value", "live-secret-value"),
        ("x-api-key: live-secret-value", "live-secret-value"),
        ("provider rejected sk-abcdefgh1234", "sk-abcdefgh1234"),
        ("provider rejected sk-cp-abcdefgh1234", "sk-cp-abcdefgh1234"),
    ],
)
def test_redact_text_removes_credentials(raw: str, secret: str) -> None:
    redacted = redact_text(raw)

    assert secret not in redacted
    assert "[REDACTED]" in redacted
