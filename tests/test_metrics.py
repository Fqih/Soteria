"""Tests for the metrics registry."""

from __future__ import annotations

import pytest

from avo.metrics import (
    MetricsError,
    MetricsRegistry,
    MetricsSnapshot,
)


def test_counter_increments_by_one() -> None:
    reg = MetricsRegistry()
    assert reg.counter("requests") == 1
    assert reg.counter("requests") == 2
    assert reg.counter("requests") == 3


def test_counter_is_isolated_by_labels() -> None:
    reg = MetricsRegistry()
    reg.counter("requests", {"endpoint": "/a"})
    reg.counter("requests", {"endpoint": "/b"})
    reg.counter("requests", {"endpoint": "/a"})
    snap = reg.snapshot()
    counters = {c.labels: c.value for c in snap.counters if c.name == "requests"}
    assert counters[frozenset({("endpoint", "/a")})] == 2
    assert counters[frozenset({("endpoint", "/b")})] == 1


def test_add_supports_custom_amounts() -> None:
    reg = MetricsRegistry()
    assert reg.add("bytes", 100) == 100
    assert reg.add("bytes", 50) == 150


def test_add_rejects_negative() -> None:
    reg = MetricsRegistry()
    with pytest.raises(MetricsError, match="non-negative"):
        reg.add("bytes", -1)


def test_gauge_records_current_value() -> None:
    reg = MetricsRegistry()
    reg.gauge("queue_depth", 10)
    reg.gauge("queue_depth", 3)
    snap = reg.snapshot()
    queue = next(g for g in snap.gauges if g.name == "queue_depth")
    assert queue.value == 3


def test_gauge_distinguishes_labels() -> None:
    reg = MetricsRegistry()
    reg.gauge("queue_depth", 10, {"worker": "a"})
    reg.gauge("queue_depth", 5, {"worker": "b"})
    snap = reg.snapshot()
    by_label = {g.labels: g.value for g in snap.gauges if g.name == "queue_depth"}
    assert by_label[frozenset({("worker", "a")})] == 10
    assert by_label[frozenset({("worker", "b")})] == 5


def test_observe_records_histogram() -> None:
    reg = MetricsRegistry()
    for value in (0.001, 0.05, 0.2, 1.5, 3.0, 9.0):
        reg.observe("latency", value)
    snap = reg.snapshot()
    hist = snap.histograms[0]
    assert hist.name == "latency"
    assert hist.count == 6
    assert hist.sum == pytest.approx(13.751, rel=1e-3)


def test_observe_respects_custom_buckets() -> None:
    reg = MetricsRegistry()
    reg.observe("custom", 5, buckets=(1.0, 10.0, 100.0))
    reg.observe("custom", 50, buckets=(1.0, 10.0, 100.0))
    reg.observe("custom", 500, buckets=(1.0, 10.0, 100.0))
    snap = reg.snapshot()
    hist = snap.histograms[0]
    assert hist.count == 3
    # cumulative: bucket counts observations ≤ upper
    assert hist.buckets == ((1.0, 0), (10.0, 1), (100.0, 2))


def test_snapshot_includes_all_kinds() -> None:
    reg = MetricsRegistry()
    reg.counter("a")
    reg.gauge("b", 1.0)
    reg.observe("c", 0.1)
    snap = reg.snapshot()
    assert isinstance(snap, MetricsSnapshot)
    assert any(s.name == "a" for s in snap.counters)
    assert any(s.name == "b" for s in snap.gauges)
    assert any(s.name == "c" for s in snap.histograms)


def test_reset_clears_state() -> None:
    reg = MetricsRegistry()
    reg.counter("a")
    reg.gauge("b", 1.0)
    reg.reset()
    snap = reg.snapshot()
    assert snap.counters == ()
    assert snap.gauges == ()


def test_cardinality_cap_raises() -> None:
    reg = MetricsRegistry(max_cardinality=2)
    reg.counter("c", {"k": "1"})
    reg.counter("c", {"k": "2"})
    with pytest.raises(MetricsError, match="cardinality"):
        reg.counter("c", {"k": "3"})


def test_invalid_max_cardinality() -> None:
    with pytest.raises(MetricsError, match="max_cardinality"):
        MetricsRegistry(max_cardinality=0)


def test_label_value_must_be_string() -> None:
    reg = MetricsRegistry()
    with pytest.raises(MetricsError, match="label values must be strings"):
        reg.counter("c", {"k": 1})  # type: ignore[dict-item]


def test_label_key_must_be_string() -> None:
    reg = MetricsRegistry()
    with pytest.raises(MetricsError, match="label keys"):
        reg.counter("c", {1: "x"})  # type: ignore[dict-item]


def test_observation_bucket_distribution() -> None:
    reg = MetricsRegistry()
    reg.observe("latency", 0.01)
    reg.observe("latency", 0.04)
    reg.observe("latency", 0.3)
    snap = reg.snapshot()
    hist = snap.histograms[0]
    by_count = {upper: count for upper, count in hist.buckets}
    assert by_count[0.005] == 0
    assert by_count[0.01] == 1
    assert by_count[0.05] == 2
    assert by_count[0.5] == 3
    assert by_count[10.0] == 3
