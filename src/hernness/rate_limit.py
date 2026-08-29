"""Async token-bucket rate limiter.

Throttles provider calls so the runtime never exceeds a configured
requests-per-minute or tokens-per-minute budget. Two independent buckets
fire on the same wall clock — request bucket ensures call frequency,
token bucket ensures token throughput.

Use as an async context manager around a provider call:

    async with limiter.acquire(input_tokens=estimate_tokens(prompt)):
        response = await provider.generate(request)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from hernness.exceptions import SoteriaError

RateLimitError = SoteriaError


@dataclass(frozen=True)
class RateLimitConfig:
    """Bucket configuration."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    def __post_init__(self) -> None:
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise RateLimitError("requests_per_minute must be positive")
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            raise RateLimitError("tokens_per_minute must be positive")

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute is not None or self.tokens_per_minute is not None


class _Bucket:
    __slots__ = ("_capacity", "_last", "_lock", "_refill_per_sec", "_tokens")

    def __init__(self, capacity: int, per_minute: int) -> None:
        self._capacity = float(capacity)
        self._refill_per_sec = per_minute / 60.0
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_sec)
            self._last = now

    async def take(self, amount: float) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                # need to wait for refill
                deficit = amount - self._tokens
                wait = deficit / self._refill_per_sec
            await asyncio.sleep(wait)


class RateLimiter:
    """Combined request + token bucket."""

    __slots__ = ("_config", "_request_bucket", "_token_bucket")

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        cfg = config or RateLimitConfig()
        self._config = cfg
        self._request_bucket: _Bucket | None = (
            _Bucket(capacity=1, per_minute=cfg.requests_per_minute)
            if cfg.requests_per_minute is not None
            else None
        )
        self._token_bucket: _Bucket | None = (
            _Bucket(capacity=cfg.tokens_per_minute, per_minute=cfg.tokens_per_minute)
            if cfg.tokens_per_minute is not None
            else None
        )

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    async def acquire(self, *, input_tokens: int = 0) -> None:
        """Block until one request + ``input_tokens`` are available."""

        if not self._config.enabled:
            return
        coros = []
        if self._request_bucket is not None:
            coros.append(self._request_bucket.take(1.0))
        if self._token_bucket is not None and input_tokens > 0:
            coros.append(self._token_bucket.take(float(input_tokens)))
        if coros:
            await asyncio.gather(*coros)


__all__ = ["RateLimitConfig", "RateLimitError", "RateLimiter"]
