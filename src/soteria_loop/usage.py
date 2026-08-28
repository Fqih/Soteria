"""Token usage tracking + cost estimation.

The runtime appends a :class:`UsageRecord` per provider call. A record
captures which step consumed what; an aggregator returns totals plus
an optional cost estimate derived from the per-1K-token rates in
``SOTERIA_USAGE_RATES`` (``SOTERIA_USAGE_RATES_INPUT_PER_1K`` and
``SOTERIA_USAGE_RATES_OUTPUT_PER_1K``).

Costs are reported in USD with up to six decimal places; missing rate
information yields a cost of ``None`` so callers can distinguish "no
usage" from "no price".
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from soteria_loop.models import TokenUsage

_DEFAULT_INPUT_RATE = Decimal("0")
_DEFAULT_OUTPUT_RATE = Decimal("0")


@dataclass(frozen=True)
class UsageRecord:
    """One provider call's usage."""

    step: int
    run_id: str
    usage: TokenUsage
    model: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "step": self.step,
            "run_id": self.run_id,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "model": self.model,
        }


def _rates_from_env() -> tuple[Decimal, Decimal]:
    """Return ``(input_per_1k, output_per_1k)`` parsed from env vars."""

    raw_in = os.environ.get("SOTERIA_USAGE_RATES_INPUT_PER_1K", "").strip()
    raw_out = os.environ.get("SOTERIA_USAGE_RATES_OUTPUT_PER_1K", "").strip()
    try:
        in_rate = Decimal(raw_in) if raw_in else _DEFAULT_INPUT_RATE
    except Exception:  # bad env never crashes runtime
        in_rate = _DEFAULT_INPUT_RATE
    try:
        out_rate = Decimal(raw_out) if raw_out else _DEFAULT_OUTPUT_RATE
    except Exception:
        out_rate = _DEFAULT_OUTPUT_RATE
    return in_rate, out_rate


def estimate_cost(
    usage: TokenUsage, *, rates: tuple[Decimal, Decimal] | None = None
) -> Decimal | None:
    """Return USD cost estimate or ``None`` when rates are unavailable."""

    in_rate, out_rate = rates if rates is not None else _rates_from_env()
    if in_rate == 0 and out_rate == 0:
        return None
    cost = (
        Decimal(usage.input_tokens) / Decimal(1000) * in_rate
        + Decimal(usage.output_tokens) / Decimal(1000) * out_rate
    )
    return cost.quantize(Decimal("0.000001"))


class UsageTracker:
    """In-memory accumulator of :class:`UsageRecord` entries."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def record(self, record: UsageRecord) -> None:
        self._records.append(record)

    def records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._records)

    def total(self) -> TokenUsage:
        running = TokenUsage()
        for record in self._records:
            running = running.plus(record.usage)
        return running

    def by_model(self) -> dict[str | None, TokenUsage]:
        out: dict[str | None, TokenUsage] = {}
        for record in self._records:
            out[record.model] = out.get(record.model, TokenUsage()).plus(record.usage)
        return out

    def cost_total(self) -> Decimal | None:
        total = self.total()
        cost = estimate_cost(total)
        return cost

    def to_list(self) -> list[dict[str, int | str | None]]:
        return [record.to_dict() for record in self._records]

    def reset(self) -> None:
        self._records.clear()


def merge(records: Iterable[UsageRecord]) -> TokenUsage:
    """Fold an iterable of records into a single :class:`TokenUsage`."""

    running = TokenUsage()
    for record in records:
        running = running.plus(record.usage)
    return running


__all__ = [
    "UsageRecord",
    "UsageTracker",
    "estimate_cost",
    "merge",
]
