"""Retry policy for transient provider errors.

The runtime wraps a provider call with :class:`RetryPolicy` so a flaky
network blip doesn't terminate a run. The policy is purely declarative:

* ``max_attempts`` — including the first try (so 1 means "no retries")
* ``initial_delay`` / ``max_delay`` — exponential backoff window in seconds
* ``retry_on`` — predicate that classifies a failure as transient

The helper :func:`call_with_retry` runs the supplied coroutine, applies
the predicate on errors, and re-raises once the budget is exhausted.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for :func:`call_with_retry`."""

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.1
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be non-negative")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if self.jitter < 0:
            raise ValueError("jitter must be non-negative")


def _backoff(policy: RetryPolicy, attempt: int) -> float:
    """Return the delay (seconds) for ``attempt`` (1-indexed)."""

    base = policy.initial_delay * (policy.multiplier ** (attempt - 1))
    bounded = min(base, policy.max_delay)
    if policy.jitter == 0:
        return bounded
    spread = bounded * policy.jitter
    return max(0.0, bounded + random.uniform(-spread, spread))


def is_transient_error(exc: BaseException) -> bool:
    """Default predicate — treat ``OSError``/``asyncio.TimeoutError`` as transient."""

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return isinstance(exc, OSError)


async def call_with_retry(
    factory: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    retry_on: Callable[[BaseException], bool] = is_transient_error,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``factory`` up to ``policy.max_attempts`` times."""

    active = policy or RetryPolicy()
    attempt = 0
    while True:
        attempt += 1
        try:
            return await factory()
        except BaseException as exc:  # retry must catch all
            if attempt >= active.max_attempts or not retry_on(exc):
                raise
            await sleep(_backoff(active, attempt))


__all__ = ["RetryPolicy", "call_with_retry", "is_transient_error"]
