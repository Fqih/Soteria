"""Tests for the token ledger."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from hernness.ledger import LedgerError, TokenLedger
from hernness.models import TokenUsage


def test_record_returns_entry_with_id(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    entry = ledger.record(
        "run-1",
        step=1,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        model="claude-fable-5",
        cost_usd=Decimal("0.001"),
    )
    assert entry.id >= 1
    assert entry.run_id == "run-1"
    assert entry.usage.total_tokens == 30
    assert entry.cost_usd == Decimal("0.001")
    ledger.close()


def test_entries_returns_all_for_run(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record("run-1", step=1, usage=TokenUsage(input_tokens=10, output_tokens=20))
    ledger.record("run-1", step=2, usage=TokenUsage(input_tokens=5, output_tokens=15))
    ledger.record("run-2", step=1, usage=TokenUsage(input_tokens=1, output_tokens=2))
    entries = ledger.entries("run-1")
    assert len(entries) == 2
    assert [e.step for e in entries] == [1, 2]
    ledger.close()


def test_total_sums_tokens(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record("run-1", step=1, usage=TokenUsage(input_tokens=10, output_tokens=20))
    ledger.record("run-1", step=2, usage=TokenUsage(input_tokens=5, output_tokens=15))
    total = ledger.total("run-1")
    assert total.input_tokens == 15
    assert total.output_tokens == 35
    assert total.total_tokens == 50
    ledger.close()


def test_total_for_missing_run_is_zero(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    total = ledger.total("run-x")
    assert total.total_tokens == 0
    ledger.close()


def test_cost_total_sums_when_all_have_cost(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record(
        "run-1",
        step=1,
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        cost_usd=Decimal("0.001"),
    )
    ledger.record(
        "run-1",
        step=2,
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        cost_usd=Decimal("0.002"),
    )
    assert ledger.cost_total("run-1") == Decimal("0.003")
    ledger.close()


def test_cost_total_returns_none_when_no_costs(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record("run-1", step=1, usage=TokenUsage(input_tokens=10, output_tokens=10))
    assert ledger.cost_total("run-1") is None
    ledger.close()


def test_cost_total_ignores_missing_costs(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record(
        "run-1",
        step=1,
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        cost_usd=Decimal("0.001"),
    )
    ledger.record("run-1", step=2, usage=TokenUsage(input_tokens=10, output_tokens=10))
    assert ledger.cost_total("run-1") == Decimal("0.001")
    ledger.close()


def test_by_model_groups_by_model(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record(
        "run-1",
        step=1,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        model="claude-fable-5",
    )
    ledger.record(
        "run-1",
        step=2,
        usage=TokenUsage(input_tokens=5, output_tokens=15),
        model="claude-haiku-4-5",
    )
    ledger.record(
        "run-1",
        step=3,
        usage=TokenUsage(input_tokens=1, output_tokens=2),
        model="claude-fable-5",
    )
    by_model = ledger.by_model("run-1")
    assert by_model["claude-fable-5"].total_tokens == 33
    assert by_model["claude-haiku-4-5"].total_tokens == 20
    ledger.close()


def test_by_model_includes_unmodeled(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record("run-1", step=1, usage=TokenUsage(input_tokens=10, output_tokens=20))
    by_model = ledger.by_model("run-1")
    assert None in by_model
    ledger.close()


def test_truncate_removes_only_target_run(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.record("run-1", step=1, usage=TokenUsage(input_tokens=10, output_tokens=20))
    ledger.record("run-2", step=1, usage=TokenUsage(input_tokens=5, output_tokens=5))
    deleted = ledger.truncate("run-1")
    assert deleted == 1
    assert ledger.entries("run-1") == ()
    assert len(ledger.entries("run-2")) == 1
    ledger.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.close()
    ledger.close()
    assert ledger._closed is True  # type: ignore[attr-defined]


def test_operations_after_close_raise(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    ledger.close()
    with pytest.raises(LedgerError, match="closed"):
        ledger.record("run-1", step=1, usage=TokenUsage())


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "ledger.db"
    ledger = TokenLedger(nested)
    ledger.record("run-1", step=1, usage=TokenUsage())
    ledger.close()
    assert nested.exists()


def test_corrupted_cost_falls_back_to_none(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    ledger = TokenLedger(path)
    ledger.record(
        "run-1",
        step=1,
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        cost_usd=Decimal("0.001"),
    )
    ledger.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE ledger_entries SET cost_usd = ? WHERE run_id = 'run-1'", ("not-a-number",))
    conn.commit()
    conn.close()
    ledger2 = TokenLedger(path)
    entry = ledger2.entries("run-1")[0]
    assert entry.cost_usd is None
    ledger2.close()


def test_to_dict_includes_all_fields(tmp_path: Path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.db")
    entry = ledger.record(
        "run-1",
        step=3,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        model="claude-fable-5",
        cost_usd=Decimal("0.005"),
    )
    out = entry.to_dict()
    assert out["run_id"] == "run-1"
    assert out["step"] == 3
    assert out["model"] == "claude-fable-5"
    assert out["input_tokens"] == 10
    assert out["cost_usd"] == "0.005"
    ledger.close()
