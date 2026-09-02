"""Circuit breaker for provider calls.

Wraps a flaky upstream (LLM API) so a sustained outage does not hammer
the endpoint and waste the run's step budget. The breaker has three
states:

* ``CLOSED`` — calls flow through. Consecutive failures are counted.
* ``OPEN`` — calls short-circuit with :class:`BreakerOpen`. After
  ``cooldown_seconds`` the breaker transitions to ``HALF_OPEN``.
* ``HALF_OPEN`` — the first ``half_open_max_calls`` calls are allowed
  through as probes. A success closes the breaker; a failure re-opens
  it for another ``cooldown_seconds``.

The breaker is intentionally simple: a single shared instance per
runtime keeps the loop deterministic and avoids hidden global state.
The runtime records success/failure in :func:`avo.runtime_handlers`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CircuitState(StrEnum):
    """State of a :class:`CircuitBreaker`."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BreakerOpen(Exception):
    """Raised by :meth:`CircuitBreaker.allow` when the breaker is rejecting calls."""

    def __init__(self, state: CircuitState, retry_after_seconds: float) -> None:
        self.state = state
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(
            f"circuit breaker {state.value}; retry after {self.retry_after_seconds:.2f}s"
        )


class CircuitBreakerPolicy(BaseModel):
    """Tuning knobs for :class:`CircuitBreaker`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_threshold: int = Field(default=5, ge=1)
    cooldown_seconds: float = Field(default=30.0, gt=0)
    half_open_max_calls: int = Field(default=1, ge=1)


class CircuitBreaker:
    """Three-state breaker — see module docstring."""

    def __init__(
        self,
        policy: CircuitBreakerPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind policy and an injectable monotonic clock for tests."""

        self.policy = policy
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._clock = clock

    @property
    def state(self) -> CircuitState:
        """Return the current state, transitioning OPEN to HALF_OPEN if cooldown elapsed."""

        self._maybe_recover()
        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failures recorded in the current CLOSED window."""

        return self._consecutive_failures

    def allow(self) -> None:
        """Reserve a slot for one call. Raises :class:`BreakerOpen` when saturated."""

        self._maybe_recover()
        if self._state is CircuitState.OPEN:
            assert self._opened_at is not None
            raise BreakerOpen(
                self._state,
                self.policy.cooldown_seconds - (self._clock() - self._opened_at),
            )
        if self._state is CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.policy.half_open_max_calls:
                raise BreakerOpen(self._state, retry_after_seconds=0.0)
            self._half_open_calls += 1

    def record_success(self) -> None:
        """Mark one call as successful; closes the breaker from HALF_OPEN."""

        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_calls = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Mark one call as failed; opens the breaker after the threshold."""

        if self._state is CircuitState.HALF_OPEN:
            self._half_open_calls = max(0, self._half_open_calls - 1)
            self._open()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.policy.failure_threshold:
            self._open()

    def reset(self) -> None:
        """Force the breaker back to CLOSED (test/admin use)."""

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_calls = 0

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._half_open_calls = 0

    def _maybe_recover(self) -> None:
        if self._state is not CircuitState.OPEN:
            return
        assert self._opened_at is not None
        if self._clock() - self._opened_at >= self.policy.cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._consecutive_failures = 0


__all__ = [
    "BreakerOpen",
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "CircuitState",
]
