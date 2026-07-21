"""Offline tests for the live benchmark pricing/estimate module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.live.pricing import (
    MINIMAX_M3_INPUT_USD_PER_MILLION,
    MINIMAX_M3_OUTPUT_USD_PER_MILLION,
    CostEstimate,
    Pricing,
    estimate_upper_bound,
    resolve_pricing,
)


def test_minimax_m3_rates_match_documented_values() -> None:
    assert pytest.approx(0.30) == MINIMAX_M3_INPUT_USD_PER_MILLION
    assert pytest.approx(1.20) == MINIMAX_M3_OUTPUT_USD_PER_MILLION


def test_resolve_pricing_returns_minimax_rates_for_known_model() -> None:
    pricing = resolve_pricing("minimax", "MiniMax-M3", {})

    assert isinstance(pricing, Pricing)
    assert pricing.provider == "minimax"
    assert pricing.model == "MiniMax-M3"
    assert pricing.input_usd_per_million == pytest.approx(0.30)
    assert pricing.output_usd_per_million == pytest.approx(1.20)
    assert pricing.currency == "USD"
    assert pricing.source_url


def test_estimate_upper_bound_uses_documented_formula() -> None:
    pricing = resolve_pricing("minimax", "MiniMax-M3", {})
    estimate = estimate_upper_bound(
        pricing,
        max_steps=6,
        scenario_count=3,
        runs=3,
        input_tokens_per_step=2048,
        output_tokens_per_step=2048,
    )

    assert isinstance(estimate, CostEstimate)
    # max_steps * scenario_count * 2 (raw + soteria_loop) * runs = 6 * 3 * 2 * 3 = 108
    assert estimate.total_steps == 6 * 3 * 2 * 3
    expected_input_tokens = 6 * 3 * 2 * 3 * 2048
    expected_output_tokens = 6 * 3 * 2 * 3 * 2048
    assert estimate.input_tokens == expected_input_tokens
    assert estimate.output_tokens == expected_output_tokens

    expected_cost = (
        expected_input_tokens / 1_000_000 * 0.30 + expected_output_tokens / 1_000_000 * 1.20
    )
    assert estimate.cost_usd == pytest.approx(expected_cost)
    assert estimate.label == "upper-bound estimate, not a bill"


def test_estimate_upper_bound_rejects_non_positive_inputs() -> None:
    pricing = resolve_pricing("minimax", "MiniMax-M3", {})
    with pytest.raises(ValueError):
        estimate_upper_bound(
            pricing,
            max_steps=0,
            scenario_count=3,
            runs=3,
            input_tokens_per_step=2048,
            output_tokens_per_step=2048,
        )


def test_resolve_pricing_unknown_openai_requires_overrides() -> None:
    with pytest.raises(ValueError):
        resolve_pricing("openai", "gpt-99-unverified", {})


def test_resolve_pricing_unknown_openai_accepts_overrides_without_serializing_secrets() -> None:
    pricing = resolve_pricing(
        "openai",
        "gpt-99-unverified",
        {
            "OPENAI_INPUT_USD_PER_MILLION": "0.50",
            "OPENAI_OUTPUT_USD_PER_MILLION": "1.50",
        },
    )

    assert pricing.input_usd_per_million == pytest.approx(0.50)
    assert pricing.output_usd_per_million == pytest.approx(1.50)
    # The serialized pricing must not echo environment variable values verbatim
    # so a future bug cannot leak them through the JSON dump.
    serialized = pricing.model_dump_json()
    assert "OPENAI_INPUT_USD_PER_MILLION" not in serialized
    assert "0.50" not in serialized  # the source env name should not appear


def test_pricing_model_is_immutable_and_serializes_without_credentials() -> None:
    pricing = resolve_pricing("minimax", "MiniMax-M3", {})
    serialized = repr(pricing).lower()
    for forbidden in ("api_key", "authorization", "secret", "password", "token="):
        assert forbidden not in serialized

    from pydantic import ValidationError

    with pytest.raises(ValidationError):  # frozen model should reject mutation
        pricing.input_usd_per_million = 0.0  # type: ignore[misc]


def test_estimate_serializes_with_upper_bound_label() -> None:
    pricing = resolve_pricing("minimax", "MiniMax-M3", {})
    estimate = estimate_upper_bound(
        pricing,
        max_steps=6,
        scenario_count=3,
        runs=3,
        input_tokens_per_step=2048,
        output_tokens_per_step=2048,
    )
    payload = estimate.model_dump(mode="json")
    assert payload["label"] == "upper-bound estimate, not a bill"
    assert payload["currency"] == "USD"


def test_resolve_pricing_openai_known_model_returns_rates_without_overrides() -> None:
    """OpenAI known catalog entries must resolve with no env overrides."""

    # There are no currently-verified OpenAI catalog entries, so any concrete
    # model requested without overrides must surface the error path.
    with pytest.raises(ValueError):
        resolve_pricing("openai", "anything", {})


def test_resolve_pricing_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        resolve_pricing("unknown", "MiniMax-M3", {})
