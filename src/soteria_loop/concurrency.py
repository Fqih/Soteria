"""Concurrency limiter for parallel tool calls.

Caps the number of in-flight tool calls so the runtime never overwhelms
an external system. Built on :class:`asyncio.Semaphore` for backpressure
plus a :class:`MetricsRegistry` counter for observability.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from soteria_loop.exceptions import SoteriaError
from soteria_loop.metrics import MetricsRegistry

ConcurrencyError = SoteriaError

T = TypeVar("T")


class ConcurrencyLimiter:
    """Async semaphore with metrics hook."""

    __slots__ = ("_in_flight_name", "_max", "_metrics", "_sem")

    def __init__(
        self,
        max_in_flight: int,
        *,
        metrics: MetricsRegistry | None = None,
        in_flight_metric: str = "concurrency.in_flight",
    ) -> None:
        if max_in_flight < 1:
            raise ConcurrencyError("max_in_flight must be >= 1")
        self._max = max_in_flight
        self._sem = asyncio.Semaphore(max_in_flight)
        self._metrics = metrics
        self._in_flight_name = in_flight_metric

    @property
    def max_in_flight(self) -> int:
        return self._max

    @property
    def available(self) -> int:
        return self._sem._value

    async def acquire(self) -> None:
        await self._sem.acquire()
        if self._metrics is not None:
            self._metrics.gauge(self._in_flight_name, float(self._max - self.available))

    def release(self) -> None:
        self._sem.release()
        if self._metrics is not None:
            self._metrics.gauge(self._in_flight_name, float(self._max - self.available))

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    async def run(self, factory: Callable[[], Awaitable[T]]) -> T:
        """Run ``factory`` under the limiter; release on completion or error."""

        async with self:
            return await factory()


__all__ = ["ConcurrencyError", "ConcurrencyLimiter"]
