"""Tests for the rate limiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from avo.rate_limit import RateLimitConfig, RateLimiter, RateLimitError


def test_config_defaults_to_disabled() -> None:
    config = RateLimitConfig()
    assert not config.enabled


def test_config_rejects_zero_rpm() -> None:
    with pytest.raises(RateLimitError, match="requests_per_minute"):
        RateLimitConfig(requests_per_minute=0)


def test_config_rejects_zero_tpm() -> None:
    with pytest.raises(RateLimitError, match="tokens_per_minute"):
        RateLimitConfig(tokens_per_minute=0)


def test_config_accepts_valid_values() -> None:
    config = RateLimitConfig(requests_per_minute=60, tokens_per_minute=100_000)
    assert config.enabled


async def test_acquire_disabled_is_noop() -> None:
    limiter = RateLimiter()
    start = time.monotonic()
    await limiter.acquire(input_tokens=1000)
    elapsed = time.monotonic() - start
    assert elapsed < 0.05


async def test_acquire_request_bucket_blocks_when_exhausted() -> None:
    config = RateLimitConfig(requests_per_minute=60)  # 1 req/sec
    limiter = RateLimiter(config)
    start = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()  # second call waits ~1 sec
    elapsed = time.monotonic() - start
    assert elapsed >= 0.9


async def test_acquire_token_bucket_blocks_until_refill() -> None:
    config = RateLimitConfig(tokens_per_minute=60)  # 1 token/sec
    limiter = RateLimiter(config)
    start = time.monotonic()
    await limiter.acquire(input_tokens=60)  # drain full bucket
    await limiter.acquire(input_tokens=1)  # waits for refill
    elapsed = time.monotonic() - start
    assert elapsed >= 0.9


async def test_acquire_combined_buckets() -> None:
    config = RateLimitConfig(requests_per_minute=6000, tokens_per_minute=600_000)
    limiter = RateLimiter(config)
    start = time.monotonic()
    await asyncio.gather(*(limiter.acquire(input_tokens=10) for _ in range(5)))
    elapsed = time.monotonic() - start
    assert elapsed < 0.2  # high limits — should not block


async def test_acquire_zero_tokens_skips_token_bucket() -> None:
    config = RateLimitConfig(tokens_per_minute=60)
    limiter = RateLimiter(config)
    start = time.monotonic()
    await limiter.acquire(input_tokens=0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.05


async def test_acquire_concurrent_calls_respect_limit() -> None:
    config = RateLimitConfig(requests_per_minute=120)  # 2 req/sec
    limiter = RateLimiter(config)
    start = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(3)))
    elapsed = time.monotonic() - start
    # capacity 2, then refill ~2/sec. third call should wait > 0.4s
    assert elapsed >= 0.4


def test_limiter_with_default_config() -> None:
    limiter = RateLimiter()
    assert not limiter.config.enabled
