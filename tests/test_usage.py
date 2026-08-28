"""Tests for the token usage tracker."""

from __future__ import annotations

from decimal import Decimal

import pytest

from soteria_loop.models import TokenUsage
from soteria_loop.usage import UsageRecord, UsageTracker, estimate_cost, merge


def _record(step: int, *, inp: int = 0, outp: int = 0, model: str | None = None) -> UsageRecord:
    return UsageRecord(
        step=step,
        run_id="run-1",
        usage=TokenUsage(input_tokens=inp, output_tokens=outp),
        model=model,
    )


def test_tracker_total_sums_records() -> None:
    tracker = UsageTracker()
    tracker.record(_record(1, inp=10, outp=5))
    tracker.record(_record(2, inp=20, outp=15))
    total = tracker.total()
    assert total.input_tokens == 30
    assert total.output_tokens == 20


def test_tracker_by_model_groups() -> None:
    tracker = UsageTracker()
    tracker.record(_record(1, inp=10, model="a"))
    tracker.record(_record(2, inp=20, model="b"))
    tracker.record(_record(3, inp=5, model="a"))
    grouped = tracker.by_model()
    assert grouped["a"].input_tokens == 15
    assert grouped["b"].input_tokens == 20


def test_tracker_by_model_handles_unknown_model() -> None:
    tracker = UsageTracker()
    tracker.record(_record(1, inp=5))  # model=None
    grouped = tracker.by_model()
    assert grouped[None].input_tokens == 5


def test_tracker_records_returns_tuple_copy() -> None:
    tracker = UsageTracker()
    tracker.record(_record(1, inp=1))
    records = tracker.records()
    assert len(records) == 1
    assert isinstance(records, tuple)


def test_tracker_reset_clears() -> None:
    tracker = UsageTracker()
    tracker.record(_record(1, inp=5))
    tracker.reset()
    assert tracker.records() == ()


def test_tracker_to_list_serialises() -> None:
    tracker = UsageTracker()
    tracker.record(_record(1, inp=5, outp=3))
    listing = tracker.to_list()
    assert listing[0]["input_tokens"] == 5
    assert listing[0]["total_tokens"] == 8


def test_estimate_cost_returns_none_without_rates() -> None:
    assert estimate_cost(TokenUsage(input_tokens=100, output_tokens=50)) is None


def test_estimate_cost_computes_usd() -> None:
    cost = estimate_cost(
        TokenUsage(input_tokens=1000, output_tokens=500),
        rates=(Decimal("0.01"), Decimal("0.03")),
    )
    assert cost == Decimal("0.025000")


def test_estimate_cost_handles_zero_tokens() -> None:
    cost = estimate_cost(TokenUsage(), rates=(Decimal("0.01"), Decimal("0.03")))
    assert cost == Decimal("0.000000")


def test_estimate_cost_uses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOTERIA_USAGE_RATES_INPUT_PER_1K", "0.005")
    monkeypatch.setenv("SOTERIA_USAGE_RATES_OUTPUT_PER_1K", "0.015")
    cost = estimate_cost(TokenUsage(input_tokens=2000, output_tokens=1000))
    assert cost == Decimal("0.025000")


def test_estimate_cost_handles_bad_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOTERIA_USAGE_RATES_INPUT_PER_1K", "not-a-number")
    monkeypatch.setenv("SOTERIA_USAGE_RATES_OUTPUT_PER_1K", "")
    assert estimate_cost(TokenUsage(input_tokens=10)) is None


def test_merge_folds_records() -> None:
    total = merge([_record(1, inp=1), _record(2, inp=2)])
    assert total.input_tokens == 3


def test_record_to_dict_includes_all_fields() -> None:
    record = _record(1, inp=2, outp=3, model="x")
    as_dict = record.to_dict()
    assert as_dict["step"] == 1
    assert as_dict["model"] == "x"
    assert as_dict["total_tokens"] == 5
