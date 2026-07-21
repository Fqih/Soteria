"""Fake-provider replay, cursor, and snapshot validation."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue

from soteria_loop.exceptions import ProviderError
from soteria_loop.models import ModelRequest, ModelResponse, TokenUsage
from soteria_loop.providers import FakeProvider


def request(step: int = 1) -> ModelRequest:
    """Build a minimal provider request."""

    return ModelRequest(run_id="fake-run", step=step, messages=[])


def test_repeat_last_requires_a_script() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FakeProvider([], repeat_last=True)


@pytest.mark.asyncio
async def test_repeat_last_cursor_and_reset_are_deterministic() -> None:
    provider = FakeProvider([ModelResponse(content="same")], repeat_last=True)

    first = await provider.generate(request())
    second = await provider.generate(request(2))
    assert first.content == second.content == "same"
    assert provider.cursor == 1
    assert len(provider.requests) == 2

    provider.reset()
    assert provider.cursor == 0
    assert provider.requests == []


@pytest.mark.asyncio
async def test_snapshot_round_trip_supports_response_mapping_and_error() -> None:
    original = FakeProvider(
        [
            ModelResponse(content="model", usage=TokenUsage(input_tokens=1)),
            {"content": "mapping"},
            RuntimeError("scripted error"),
        ]
    )
    restored = FakeProvider.from_snapshot(original.snapshot_state())

    assert (await restored.generate(request())).content == "model"
    assert (await restored.generate(request(2))).content == "mapping"
    with pytest.raises(ProviderError, match="scripted error"):
        await restored.generate(request(3))


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"provider_type": "other"}, "another provider"),
        ({"provider_type": "fake"}, "script list"),
        (
            {
                "provider_type": "fake",
                "script": ["bad"],
                "cursor": 0,
                "repeat_last": False,
            },
            "invalid script entry",
        ),
        (
            {
                "provider_type": "fake",
                "script": [{"kind": "mapping", "value": "bad"}],
                "cursor": 0,
                "repeat_last": False,
            },
            "object value",
        ),
        (
            {
                "provider_type": "fake",
                "script": [{"kind": "unknown"}],
                "cursor": 0,
                "repeat_last": False,
            },
            "Unknown",
        ),
        (
            {
                "provider_type": "fake",
                "script": [],
                "cursor": -1,
                "repeat_last": False,
            },
            "non-negative",
        ),
        (
            {
                "provider_type": "fake",
                "script": [],
                "cursor": 0,
                "repeat_last": "no",
            },
            "boolean",
        ),
        (
            {
                "provider_type": "fake",
                "script": [],
                "cursor": 1,
                "repeat_last": False,
            },
            "exceeds",
        ),
        (
            {
                "provider_type": "fake",
                "script": [],
                "cursor": 0,
                "repeat_last": True,
            },
            "empty script",
        ),
    ],
)
def test_snapshot_validation_is_actionable(
    state: object,
    message: str,
) -> None:
    provider = FakeProvider([])

    with pytest.raises(ValueError, match=message):
        provider.restore_state(cast(dict[str, JsonValue], state))
