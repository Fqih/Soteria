"""In-process metrics registry.

Three metric kinds:

* Counter — monotonic, incremented on each event.
* Gauge — current value, may go up or down.
* Histogram — bucketed distribution, increments a counter bucket.

All metrics are identified by ``(name, labels)`` where ``labels`` is an
immutable mapping of string keys to string values. The registry caps
label-cardinality per metric so a runaway label combination can never
OOM the process.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hernness.exceptions import SoteriaError

MetricsError = SoteriaError

_DEFAULT_HISTOGRAM_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _validate_labels(labels: Mapping[str, str]) -> None:
    for key in labels:
        if not isinstance(key, str) or not key:
            raise MetricsError("label keys must be non-empty strings")
    for value in labels.values():
        if not isinstance(value, str):
            raise MetricsError("label values must be strings")


@dataclass(frozen=True)
class CounterSnapshot:
    """Point-in-time view of a counter."""

    name: str
    labels: frozenset[tuple[str, str]]
    value: int


@dataclass(frozen=True)
class GaugeSnapshot:
    """Point-in-time view of a gauge."""

    name: str
    labels: frozenset[tuple[str, str]]
    value: float


@dataclass(frozen=True)
class HistogramSnapshot:
    """Point-in-time view of a histogram."""

    name: str
    labels: frozenset[tuple[str, str]]
    count: int
    sum: float
    buckets: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class MetricsSnapshot:
    counters: tuple[CounterSnapshot, ...]
    gauges: tuple[GaugeSnapshot, ...]
    histograms: tuple[HistogramSnapshot, ...]


class _CounterCell:
    __slots__ = ("labels", "value")

    def __init__(self, labels: frozenset[tuple[str, str]]) -> None:
        self.labels = labels
        self.value = 0


class _GaugeCell:
    __slots__ = ("labels", "value")

    def __init__(self, labels: frozenset[tuple[str, str]], value: float = 0.0) -> None:
        self.labels = labels
        self.value = value


class _HistogramCell:
    __slots__ = ("buckets", "counts", "labels", "sum", "total")

    def __init__(
        self,
        labels: frozenset[tuple[str, str]],
        buckets: tuple[float, ...],
    ) -> None:
        self.labels = labels
        self.buckets = buckets
        self.counts: list[int] = [0] * len(buckets)
        self.sum = 0.0
        self.total = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.total += 1
        for i in range(len(self.buckets) - 1, -1, -1):
            if value <= self.buckets[i]:
                self.counts[i] += 1
            else:
                break


class MetricsRegistry:
    """Thread-safe metric store."""

    __slots__ = ("_counters", "_gauges", "_histograms", "_max_cardinality")

    def __init__(self, *, max_cardinality: int = 1024) -> None:
        if max_cardinality < 1:
            raise MetricsError("max_cardinality must be >= 1")
        self._max_cardinality = max_cardinality
        self._counters: dict[str, dict[frozenset[tuple[str, str]], _CounterCell]] = defaultdict(
            dict
        )
        self._gauges: dict[str, dict[frozenset[tuple[str, str]], _GaugeCell]] = defaultdict(dict)
        self._histograms: dict[str, dict[frozenset[tuple[str, str]], _HistogramCell]] = defaultdict(
            dict
        )

    def _check_cardinality(self, store: Mapping[Any, Any]) -> None:
        if len(store) >= self._max_cardinality:
            raise MetricsError(f"metric exceeded cardinality cap ({self._max_cardinality})")

    @staticmethod
    def _label_key(labels: Mapping[str, str]) -> frozenset[tuple[str, str]]:
        _validate_labels(labels)
        return frozenset(labels.items())

    def counter(self, name: str, labels: Mapping[str, str] | None = None) -> int:
        key = self._label_key(labels or {})
        store = self._counters[name]
        cell = store.get(key)
        if cell is None:
            self._check_cardinality(store)
            cell = _CounterCell(key)
            store[key] = cell
        cell.value += 1
        return cell.value

    def add(self, name: str, amount: int, labels: Mapping[str, str] | None = None) -> int:
        if amount < 0:
            raise MetricsError("counter increments must be non-negative")
        key = self._label_key(labels or {})
        store = self._counters[name]
        cell = store.get(key)
        if cell is None:
            self._check_cardinality(store)
            cell = _CounterCell(key)
            store[key] = cell
        cell.value += amount
        return cell.value

    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> float:
        key = self._label_key(labels or {})
        store = self._gauges[name]
        cell = store.get(key)
        if cell is None:
            self._check_cardinality(store)
            cell = _GaugeCell(key, value)
            store[key] = cell
        else:
            cell.value = value
        return cell.value

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        buckets: tuple[float, ...] = _DEFAULT_HISTOGRAM_BUCKETS,
    ) -> None:
        key = self._label_key(labels or {})
        store = self._histograms[name]
        cell = store.get(key)
        if cell is None:
            self._check_cardinality(store)
            cell = _HistogramCell(key, buckets)
            store[key] = cell
        cell.observe(value)

    def snapshot(self) -> MetricsSnapshot:
        counters = tuple(
            CounterSnapshot(name=name, labels=cell.labels, value=cell.value)
            for name, store in self._counters.items()
            for cell in store.values()
        )
        gauges = tuple(
            GaugeSnapshot(name=name, labels=cell.labels, value=cell.value)
            for name, store in self._gauges.items()
            for cell in store.values()
        )
        histograms = tuple(
            HistogramSnapshot(
                name=name,
                labels=cell.labels,
                count=cell.total,
                sum=cell.sum,
                buckets=tuple(zip(cell.buckets, cell.counts, strict=True)),
            )
            for name, store in self._histograms.items()
            for cell in store.values()
        )
        return MetricsSnapshot(counters=counters, gauges=gauges, histograms=histograms)

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


__all__ = [
    "CounterSnapshot",
    "GaugeSnapshot",
    "HistogramSnapshot",
    "MetricsError",
    "MetricsRegistry",
    "MetricsSnapshot",
]
