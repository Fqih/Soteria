"""Tests for the Groq + Cerebras provider adapters."""

from __future__ import annotations

from typing import Any

import pytest

from avo.providers.cerebras import CerebrasConfig, CerebrasProvider
from avo.providers.groq import GroqConfig, GroqProvider


class _StubClient:
    """Captures the request that would be sent over the wire."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_payload: dict[str, Any] | None = None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None,  # noqa: ASYNC109 - mirrors the httpx client signature
    ) -> Any:
        del timeout
        self.last_url = url
        self.last_headers = headers
        self.last_payload = json
        return _StubResponse(self.response)

    async def aclose(self) -> None:
        pass


class _StubResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.status_code = 200
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._body


def _sample_payload() -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def test_groq_config_requires_api_key() -> None:
    env: dict[str, str] = {"AVO_GROQ_MODEL": "llama-3.3-70b-versatile"}
    with pytest.raises(ValueError, match="AVO_GROQ_API_KEY is required"):
        GroqConfig.from_avo_env(env, fallback_model="llama-3.3-70b-versatile")


def test_groq_config_reads_base_url_override() -> None:
    env = {
        "AVO_GROQ_API_KEY": "test-key",
        "AVO_GROQ_BASE_URL": "https://custom.groq.test/v1/",
        "AVO_GROQ_MODEL": "llama-3.1-8b-instant",
    }
    config = GroqConfig.from_avo_env(env, fallback_model="x")
    assert config.base_url == "https://custom.groq.test/v1"
    assert config.model == "llama-3.1-8b-instant"
    assert config.endpoint.endswith("/chat/completions")


def test_groq_provider_posts_with_bearer_auth() -> None:
    env = {"AVO_GROQ_API_KEY": "sk-test", "AVO_GROQ_MODEL": "llama-3.3-70b-versatile"}
    config = GroqConfig.from_avo_env(env, fallback_model="x")
    client = _StubClient(_sample_payload())
    provider = GroqProvider(config, client=client)  # type: ignore[arg-type]

    from avo.models import ModelRequest

    request = ModelRequest(run_id="r1", step=1, messages=[])

    async def go() -> str:
        result = await provider.generate(request)
        return result.content or ""

    import asyncio

    assert asyncio.run(go()) == "hello"
    assert client.last_url is not None
    assert client.last_url.startswith("https://api.groq.com/openai/v1/chat/completions")
    assert client.last_headers is not None
    assert client.last_headers["Authorization"] == "Bearer sk-test"


def test_cerebras_config_requires_api_key() -> None:
    env: dict[str, str] = {"AVO_CEREBRAS_MODEL": "llama-3.3-70b"}
    with pytest.raises(ValueError, match="AVO_CEREBRAS_API_KEY is required"):
        CerebrasConfig.from_avo_env(env, fallback_model="llama-3.3-70b")


def test_cerebras_provider_uses_default_base_url() -> None:
    env = {"AVO_CEREBRAS_API_KEY": "sk-cer", "AVO_CEREBRAS_MODEL": "llama-3.1-8b"}
    config = CerebrasConfig.from_avo_env(env, fallback_model="x")
    assert config.base_url == "https://api.cerebras.ai/v1"


def test_cerebras_provider_posts_to_default_endpoint() -> None:
    env = {"AVO_CEREBRAS_API_KEY": "sk-cer", "AVO_CEREBRAS_MODEL": "llama-3.3-70b"}
    config = CerebrasConfig.from_avo_env(env, fallback_model="x")
    client = _StubClient(_sample_payload())
    provider = CerebrasProvider(config, client=client)  # type: ignore[arg-type]

    from avo.models import ModelRequest

    request = ModelRequest(run_id="r1", step=1, messages=[])

    async def go() -> int:
        result = await provider.generate(request)
        usage = result.usage
        return usage.input_tokens if usage is not None else 0

    import asyncio

    assert asyncio.run(go()) == 5
    assert client.last_url is not None
    assert client.last_url.startswith("https://api.cerebras.ai/v1/chat/completions")


def test_groq_and_cerebras_have_distinct_name_attr() -> None:
    assert GroqProvider.name == "groq"
    assert CerebrasProvider.name == "cerebras"


def test_groq_provider_rejects_missing_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """When httpx is unavailable and no client is injected, generate() raises."""

    env = {"AVO_GROQ_API_KEY": "sk", "AVO_GROQ_MODEL": "x"}
    config = GroqConfig.from_avo_env(env, fallback_model="x")
    provider = GroqProvider(config, client=None)  # type: ignore[arg-type]
    provider._client = None  # force the missing-client branch

    from avo.exceptions import ProviderError
    from avo.models import ModelRequest

    request = ModelRequest(run_id="r1", step=1, messages=[])

    async def go() -> None:
        await provider.generate(request)

    import asyncio

    with pytest.raises(ProviderError, match="GroqProvider requires httpx"):
        asyncio.run(go())
