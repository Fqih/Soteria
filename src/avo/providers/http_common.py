"""Shared helpers for OpenAI-compatible live model providers."""

from __future__ import annotations

import json
import re
from typing import Any

from avo import ModelRequest, ModelResponse, TokenUsage, ToolCall
from avo.exceptions import ProviderError


def json_safe_content(value: object) -> str:
    """Return text unchanged and encode other values as deterministic JSON."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def redact_text(value: str) -> str:
    """Remove common API credentials from provider-controlled text."""

    redacted = re.sub(
        r"(?i)(authorization|x-api-key)(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1\2[REDACTED]",
        value,
    )
    return re.sub(
        r"\b(sk-[A-Za-z0-9_-]{8,}|sk-cp-[A-Za-z0-9_-]{8,})\b",
        "[REDACTED]",
        redacted,
    )


def build_openai_payload(
    model: str,
    request: ModelRequest,
    max_completion_tokens: int,
) -> dict[str, object]:
    """Convert a Avo model request to an OpenAI-compatible payload."""

    messages = [_openai_message(message) for message in request.messages]
    tools = [
        {
            "type": "function",
            "function": {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": metadata.input_schema,
            },
        }
        for metadata in request.tools
    ]
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_completion_tokens,
        "max_completion_tokens": max_completion_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def parse_openai_response(payload: object) -> ModelResponse:
    """Parse one non-streaming OpenAI-compatible response."""

    try:
        response = _require_dict(payload)
        choices = response["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices must be a non-empty list")
        choice = _require_dict(choices[0])
        message = _require_dict(choice["message"])
        usage = _parse_usage(response.get("usage"))

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            tool_call = _parse_tool_call(tool_calls)
            return ModelResponse(tool_call=tool_call, usage=usage)

        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        return ModelResponse(content=content, usage=usage)
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        detail = redact_text(str(exc))
        raise ProviderError(
            f"Invalid OpenAI-compatible response: {detail}",
            retryable=False,
        ) from exc


def _openai_message(message: dict[str, Any]) -> dict[str, object]:
    role = message["role"]
    if role == "assistant" and "tool_call" in message:
        tool_call = _require_dict(message["tool_call"])
        arguments = tool_call["arguments"]
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(arguments, sort_keys=True),
                    },
                }
            ],
        }
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message["tool_call_id"],
            "content": json_safe_content(message["content"]),
        }
    return {"role": role, "content": message["content"]}


def _parse_tool_call(value: object) -> ToolCall:
    if not isinstance(value, list) or not value:
        raise ValueError("tool_calls must be a non-empty list")
    raw_call = _require_dict(value[0])
    function = _require_dict(raw_call["function"])
    raw_arguments = function["arguments"]
    if not isinstance(raw_arguments, str):
        raise ValueError("tool call arguments must be a JSON string")
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must decode to an object")
    return ToolCall(
        tool_call_id=raw_call["id"],
        name=function["name"],
        arguments=arguments,
    )


def _parse_usage(value: object) -> TokenUsage | None:
    if value is None:
        return None
    usage = _require_dict(value)
    return TokenUsage(
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )


def _require_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    return value
