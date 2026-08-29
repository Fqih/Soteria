"""Tests for the concurrency limiter."""

from __future__ import annotations

import asyncio

import pytest

from soteria_loop.concurrency import ConcurrencyError, ConcurrencyLimiter
from soteria_loop.metrics import MetricsRegistry


def test_max_in_flight_must_be_positive() -> None:
    with pytest.raises(ConcurrencyError, match="max_in_flight"):
        ConcurrencyLimiter(0)


async def test_run_executes_under_limit() -> None:
    limiter = ConcurrencyLimiter(2)

    async def work() -> str:
        return "done"

    result = await limiter.run(work)
    assert result == "done"


async def test_run_blocks_when_saturated() -> None:
    limiter = ConcurrencyLimiter(1)
    started: list[int] = []
    finished: list[int] = []

    async def task(n: int) -> int:
        started.append(n)
        await asyncio.sleep(0.05)
        finished.append(n)
        return n

    results = await asyncio.gather(limiter.run(lambda: task(1)), limiter.run(lambda: task(2)))
    assert sorted(results) == [1, 2]
    assert started == [1, 2]


async def test_context_manager_releases_on_exit() -> None:
    limiter = ConcurrencyLimiter(1)

    async def hold() -> None:
        await limiter.acquire()
        try:
            assert limiter.available == 0
        finally:
            limiter.release()

    await hold()
    assert limiter.available == 1


async def test_context_manager_releases_on_exception() -> None:
    limiter = ConcurrencyLimiter(1)

    async def boom() -> None:
        async with limiter:
            raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await boom()
    assert limiter.available == 1


async def test_metrics_records_in_flight() -> None:
    metrics = MetricsRegistry()
    limiter = ConcurrencyLimiter(2, metrics=metrics, in_flight_metric="test.in_flight")

    async def hold() -> None:
        async with limiter:
            await asyncio.sleep(0.05)

    await asyncio.gather(hold(), hold())
    final = metrics.snapshot().gauges
    in_flight = [g for g in final if g.name == "test.in_flight"]
    assert in_flight[-1].value == 0.0


async def test_release_does_not_raise_when_unbalanced() -> None:
    # asyncio.Semaphore.release() is forgiving — it does not raise when
    # the counter would exceed the initial value. The limiter inherits
    # that behavior rather than wrapping the call.
    limiter = ConcurrencyLimiter(1)
    limiter.release()
    assert limiter.available == 2


async def test_run_many_serializes_through_single_slot() -> None:
    limiter = ConcurrencyLimiter(1)
    order: list[int] = []

    async def work(n: int) -> int:
        order.append(n)
        await asyncio.sleep(0.01)
        return n

    results = await asyncio.gather(*(limiter.run(lambda n=k: work(n)) for k in range(3)))
    assert results == [0, 1, 2]
    assert order == [0, 1, 2]


def test_max_in_flight_property() -> None:
    limiter = ConcurrencyLimiter(5)
    assert limiter.max_in_flight == 5
