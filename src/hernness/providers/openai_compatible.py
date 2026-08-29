"""OpenAI Chat Completions-compatible provider.

Targets any endpoint that speaks the OpenAI ``/chat/completions`` schema:
the real ``api.openai.com`` as well as self-hosted or third-party
OpenAI-compatible gateways (Together, vLLM, llama.cpp's server, ...).

Configuration comes from the HERNNESS_-prefixed environment documented in
:mod:`hernness.config`. The legacy ``OPENAI_*`` variables used by the
live-benchmark harness remain supported through ``OpenAIConfig.from_env``
in :mod:`hernness.providers.openai`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hernness import ModelRequest, ModelResponse
from hernness.exceptions import ProviderError

from .http_common import build_openai_payload, parse_openai_response, redact_text

try:  # pragma: no cover - exercised indirectly by the optional dependency
    import httpx
except ModuleNotFoundError:  # pragma: no cover - httpx is optional at import time
    httpx = None  # type: ignore[assignment]


_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class _AsyncHTTPClient(Protocol):
    """Minimal async client surface used by :class:`OpenAICompatibleProvider`."""

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None,  # noqa: ASYNC109 - mirrors the httpx client signature
    ) -> Any: ...

    async def aclose(self) -> None: ...


class OpenAICompatibleConfig(BaseModel):
    """Endpoint and credential configuration for an OpenAI-compatible API."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str
    base_url: str = _DEFAULT_BASE_URL

    _api_key: str | None = PrivateAttr(default=None)

    @classmethod
    def from_soteria_env(
        cls,
        environ: Mapping[str, str],
        *,
        fallback_model: str,
    ) -> OpenAICompatibleConfig:
        """Build config from the HERNNESS_OPENAI_* environment variables."""

        api_key = environ.get("HERNNESS_OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("HERNNESS_OPENAI_API_KEY is required when HERNNESS_PROVIDER=openai")

        model = environ.get("HERNNESS_OPENAI_MODEL", "").strip() or fallback_model
        base_url = (
            environ.get("HERNNESS_OPENAI_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        ).rstrip("/")

        config = cls(model=model, base_url=base_url)
        config._api_key = api_key
        return config

    @property
    def endpoint(self) -> str:
        """Return the configured chat-completions endpoint."""

        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        """Bearer authorization plus content-type."""

        return {
            "Authorization": f"Bearer {self._api_key or ''}",
            "Content-Type": "application/json",
        }


class OpenAICompatibleProvider:
    """An async ``ModelProvider`` for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
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
        """Generate one final answer or tool-call decision."""

        if self._client is None:  # pragma: no cover - requires missing httpx
            raise ProviderError(
                "OpenAICompatibleProvider requires httpx or an injected client",
                retryable=False,
            )

        payload = build_openai_payload(
            self._config.model,
            request,
            self._max_completion_tokens,
        )
        raw = await self._post(payload)
        return parse_openai_response(raw)

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
                f"OpenAI transport failure: {redact_text(str(exc))}",
                retryable=True,
            ) from exc

        status = int(response.status_code)
        if status >= 400:
            detail = redact_text(str(getattr(response, "text", "")))
            raise ProviderError(
                f"OpenAI request failed with status {status}: {detail}",
                retryable=status == 429 or status >= 500,
            )

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ProviderError(
                f"OpenAI returned an unparsable response body: {redact_text(str(exc))}",
                retryable=False,
            ) from exc

    async def aclose(self) -> None:
        """Close the underlying client when this provider created it."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()


__all__ = ["OpenAICompatibleConfig", "OpenAICompatibleProvider"]
