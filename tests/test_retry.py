"""Tests for the retry policy."""

from __future__ import annotations

import asyncio

import pytest

from avo.retry import RetryPolicy, call_with_retry, is_transient_error


def test_policy_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_policy_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="initial_delay"):
        RetryPolicy(initial_delay=-1.0)


def test_policy_rejects_max_below_initial() -> None:
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(initial_delay=2.0, max_delay=1.0)


def test_policy_rejects_bad_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        RetryPolicy(multiplier=0.0)


def test_is_transient_default_predicate() -> None:
    assert is_transient_error(ConnectionError("x"))
    assert is_transient_error(TimeoutError())
    assert is_transient_error(OSError("x"))
    assert not is_transient_error(ValueError("x"))


async def test_call_succeeds_first_attempt() -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await call_with_retry(
        factory,
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _s: asyncio.sleep(0),
    )
    assert result == "ok"
    assert calls == 1


async def test_call_retries_on_transient_then_succeeds() -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("blip")
        return "ok"

    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    result = await call_with_retry(
        factory,
        policy=RetryPolicy(max_attempts=5, initial_delay=0.5, multiplier=2.0, jitter=0.0),
        sleep=fake_sleep,
    )
    assert result == "ok"
    assert calls == 3
    assert sleeps == [0.5, 1.0]


async def test_call_gives_up_after_max_attempts() -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        raise ConnectionError("always")

    with pytest.raises(ConnectionError, match="always"):
        await call_with_retry(
            factory,
            policy=RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0),
            sleep=lambda _s: asyncio.sleep(0),
        )
    assert calls == 3


async def test_call_does_not_retry_on_non_transient() -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("hard")

    with pytest.raises(ValueError):
        await call_with_retry(
            factory,
            policy=RetryPolicy(max_attempts=5),
            sleep=lambda _s: asyncio.sleep(0),
        )
    assert calls == 1


async def test_call_uses_custom_predicate() -> None:
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("retryable")
        return "done"

    def retryable(exc: BaseException) -> bool:
        return isinstance(exc, RuntimeError)

    result = await call_with_retry(
        factory,
        policy=RetryPolicy(max_attempts=3, initial_delay=0.0, jitter=0.0),
        retry_on=retryable,
        sleep=lambda _s: asyncio.sleep(0),
    )
    assert result == "done"
    assert calls == 2


async def test_backoff_caps_at_max_delay() -> None:
    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    async def factory() -> None:
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        await call_with_retry(
            factory,
            policy=RetryPolicy(
                max_attempts=5,
                initial_delay=1.0,
                max_delay=2.0,
                multiplier=10.0,
                jitter=0.0,
            ),
            sleep=fake_sleep,
        )
    # Exponential: 1, 10→cap 2, 100→cap 2, 1000→cap 2
    assert sleeps == [1.0, 2.0, 2.0, 2.0]


async def test_call_with_default_policy_succeeds() -> None:
    async def factory() -> str:
        return "ok"

    result = await call_with_retry(factory)
    assert result == "ok"
