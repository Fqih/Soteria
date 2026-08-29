"""Native Ollama chat-completions provider.

Ollama's native ``/api/chat`` endpoint accepts the same OpenAI-compatible
tool schema and returns an assistant ``message`` with optional
``tool_calls``. Token usage surfaces as ``prompt_eval_count`` and
``eval_count`` only when the response is finished, so the provider must
leave ``usage`` unset whenever those fields are absent rather than fabricate
zeros.

Configuration comes from the HERNNESS_-prefixed environment documented in
:mod:`hernness.config`. The ``AUTH_TOKEN``/``OPENAI_AUTH_TOKEN``
convention used by the legacy live-benchmark harness does not apply here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hernness import ModelRequest, ModelResponse, TokenUsage, ToolCall
from hernness.exceptions import ProviderError

from .http_common import json_safe_content, redact_text

try:  # pragma: no cover - exercised indirectly by the optional dependency
    import httpx
except ModuleNotFoundError:  # pragma: no cover - httpx is optional at import time
    httpx = None  # type: ignore[assignment]


_DEFAULT_BASE_URL = "http://localhost:11434"


class _AsyncHTTPClient(Protocol):
    """Minimal async client surface used by :class:`OllamaProvider`."""

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None,  # noqa: ASYNC109 - mirrors the httpx client signature
    ) -> Any: ...

    async def aclose(self) -> None: ...


class OllamaConfig(BaseModel):
    """Endpoint configuration for a local Ollama instance."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str
    base_url: str = _DEFAULT_BASE_URL

    _api_key: str | None = PrivateAttr(default=None)

    @classmethod
    def from_hernness_env(
        cls,
        environ: Mapping[str, str],
        *,
        fallback_model: str,
    ) -> OllamaConfig:
        """Build config from the HERNNESS_OLLAMA_* environment variables."""

        model = environ.get("HERNNESS_OLLAMA_MODEL", "").strip() or fallback_model
        base_url = (
            environ.get("HERNNESS_OLLAMA_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        ).rstrip("/")
        api_key = environ.get("HERNNESS_OLLAMA_API_KEY", "").strip() or None
        config = cls(model=model, base_url=base_url)
        config._api_key = api_key
        return config

    @property
    def endpoint(self) -> str:
        """Return the absolute native chat-completions URL."""

        return f"{self.base_url}/api/chat"

    def headers(self) -> dict[str, str]:
        """Authorization headers; Ollama itself does not require auth."""

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


class OllamaProvider:
    """An async ``ModelProvider`` for a local Ollama server."""

    def __init__(
        self,
        config: OllamaConfig,
        max_completion_tokens: int = 1024,
        request_timeout_seconds: float = 30.0,
        *,
        client: _AsyncHTTPClient | None = None,
    ) -> None:
        self._config = config
        self._max_completion_tokens = max_completion_tokens
        self._request_timeout_seconds = request_timeout_seconds
        self._owns_client = client is None
        if client is not None:
            self._client: _AsyncHTTPClient | None = client
        elif httpx is not None:
            self._client = httpx.AsyncClient(timeout=request_timeout_seconds)
        else:  # pragma: no cover - only when httpx is not installed
            self._client = None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one final answer or tool-call decision via Ollama."""

        if self._client is None:  # pragma: no cover - requires missing httpx
            raise ProviderError(
                "OllamaProvider requires httpx or an injected client",
                retryable=False,
            )

        payload = self._build_payload(request)
        raw = await self._post(payload)
        return self._parse_response(raw)

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        messages = [_ollama_message(message) for message in request.messages]
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self._max_completion_tokens},
        }
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
        if tools:
            payload["tools"] = tools
        return payload

    async def _post(self, payload: dict[str, Any]) -> Any:
        assert self._client is not None
        transport_errors: tuple[type[BaseException], ...] = (
            (httpx.HTTPError,) if httpx is not None else ()
        )
        try:
            response = await self._client.post(
                self._config.endpoint,
                headers=self._config.headers(),
                json=payload,
                timeout=self._request_timeout_seconds,
            )
        except transport_errors as exc:
            raise ProviderError(
                f"Ollama transport failure: {redact_text(str(exc))}",
                retryable=True,
            ) from exc

        status = int(response.status_code)
        if status >= 400:
            detail = redact_text(str(getattr(response, "text", "")))
            raise ProviderError(
                f"Ollama request failed with status {status}: {detail}",
                retryable=status == 429 or status >= 500,
            )

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ProviderError(
                f"Ollama returned an unparsable response body: {redact_text(str(exc))}",
                retryable=False,
            ) from exc

    @staticmethod
    def _parse_response(payload: object) -> ModelResponse:
        try:
            if not isinstance(payload, dict):
                raise TypeError("expected an object")
            message = payload["message"]
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            usage = OllamaProvider._parse_usage(payload)

            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                tool_call = OllamaProvider._parse_tool_call(tool_calls)
                return ModelResponse(tool_call=tool_call, usage=usage)

            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("message content must be a string")
            return ModelResponse(content=content, usage=usage)
        except ProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"Invalid Ollama response: {redact_text(str(exc))}",
                retryable=False,
            ) from exc

    @staticmethod
    def _parse_usage(payload: dict[str, Any]) -> TokenUsage | None:
        """Return usage only when Ollama reported both halves."""

        prompt_eval = payload.get("prompt_eval_count")
        eval_count = payload.get("eval_count")
        if not isinstance(prompt_eval, int) or not isinstance(eval_count, int):
            return None
        if prompt_eval < 0 or eval_count < 0:
            raise ValueError("token counts must be non-negative")
        return TokenUsage(input_tokens=prompt_eval, output_tokens=eval_count)

    @staticmethod
    def _parse_tool_call(value: object) -> ToolCall:
        if not isinstance(value, list) or not value:
            raise ValueError("tool_calls must be a non-empty list")
        raw_call = value[0]
        if not isinstance(raw_call, dict):
            raise ValueError("tool call must be an object")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool call.function must be an object")
        raw_arguments = function.get("arguments")
        arguments = _coerce_arguments(raw_arguments)
        return ToolCall(
            tool_call_id=str(raw_call.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=arguments,
        )

    async def aclose(self) -> None:
        """Close the underlying client when this provider created it."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()


def _ollama_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]
    if role == "assistant" and "tool_call" in message:
        tool_call = message["tool_call"]
        if not isinstance(tool_call, dict):
            raise ValueError("assistant tool_call must be an object")
        return {
            "role": "assistant",
            "content": tool_call.get("content", ""),
            "tool_calls": [
                {
                    "function": {
                        "name": tool_call.get("name"),
                        "arguments": tool_call.get("arguments", {}),
                    }
                }
            ],
        }
    if role == "tool":
        return {
            "role": "tool",
            "content": json_safe_content(message.get("content")),
        }
    return {"role": role, "content": message.get("content")}


def _coerce_arguments(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("tool call arguments must decode as JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("tool call arguments must decode to an object")
        return decoded
    raise ValueError("tool call arguments must be a string or object")


__all__ = ["OllamaConfig", "OllamaProvider"]
