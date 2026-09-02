"""Tests for :mod:`avo.circuit_breaker` and runtime integration."""

from __future__ import annotations

import pytest

from avo.circuit_breaker import (
    BreakerOpen,
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitState,
)
from avo.policies import LoopPolicy

# ---------------------------------------------------------------------------
# Pure breaker unit tests
# ---------------------------------------------------------------------------


class _Clock:
    """Monotonic clock stub controllable from tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _policy(
    *,
    failure_threshold: int = 3,
    cooldown_seconds: float = 5.0,
    half_open_max_calls: int = 1,
) -> CircuitBreakerPolicy:
    return CircuitBreakerPolicy(
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
        half_open_max_calls=half_open_max_calls,
    )


def test_closed_lets_calls_through() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(), clock=clock)
    breaker.allow()
    breaker.allow()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_opens_after_threshold_failures() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(failure_threshold=3), clock=clock)
    breaker.allow()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_open_short_circuits_calls() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(cooldown_seconds=5.0), clock=clock)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    with pytest.raises(BreakerOpen) as exc_info:
        breaker.allow()
    assert exc_info.value.state is CircuitState.OPEN
    assert exc_info.value.retry_after_seconds > 0


def test_transitions_to_half_open_after_cooldown() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(cooldown_seconds=5.0), clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(5.0)
    assert breaker.state is CircuitState.HALF_OPEN


def test_half_open_allows_single_probe_then_rejects() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(half_open_max_calls=1), clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(5.0)
    breaker.allow()
    with pytest.raises(BreakerOpen) as exc_info:
        breaker.allow()
    assert exc_info.value.state is CircuitState.HALF_OPEN


def test_half_open_success_closes_breaker() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(half_open_max_calls=1), clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(5.0)
    breaker.allow()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
    breaker.allow()  # no raise


def test_half_open_failure_reopens_with_full_cooldown() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(cooldown_seconds=5.0), clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(5.0)
    breaker.allow()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    # Before cooldown elapses, calls are still rejected
    with pytest.raises(BreakerOpen):
        breaker.allow()
    clock.advance(5.0)
    breaker.allow()  # half-open again


def test_success_resets_consecutive_failures() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(failure_threshold=3), clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_reset_returns_to_closed() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(_policy(), clock=clock)
    for _ in range(3):
        breaker.record_failure()
    breaker.reset()
    assert breaker.state is CircuitState.CLOSED
    breaker.allow()  # no raise


# ---------------------------------------------------------------------------
# LoopPolicy integration
# ---------------------------------------------------------------------------


def test_loop_policy_circuit_breaker_default_none() -> None:
    policy = LoopPolicy()
    assert policy.circuit_breaker is None


def test_loop_policy_circuit_breaker_serialized() -> None:
    policy = LoopPolicy(circuit_breaker=CircuitBreakerPolicy(failure_threshold=2))
    dumped = policy.model_dump()
    assert dumped["circuit_breaker"]["failure_threshold"] == 2


# ---------------------------------------------------------------------------
# Runtime integration: provider failures feed the breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_opens_breaker_after_repeated_provider_failures() -> None:
    from avo import AgentRuntime, ModelRequest, ModelResponse
    from avo.exceptions import ProviderError

    class _FlakyProvider:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            raise ProviderError("boom", retryable=True)

    runtime = AgentRuntime(
        provider=_FlakyProvider(),
        policy=LoopPolicy(
            max_steps=5,
            max_runtime_seconds=10,
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=2, cooldown_seconds=60.0),
        ),
    )
    # Two consecutive failed generations should open the breaker.
    for _ in range(2):
        await runtime.run("test")
    assert runtime._breaker is not None
    assert runtime._breaker.state is CircuitState.OPEN
    # Third call short-circuits without invoking the provider at all.
    calls_before = runtime.provider.calls  # type: ignore[attr-defined]
    await runtime.run("test")
    assert runtime.provider.calls == calls_before  # type: ignore[attr-defined]
