"""Per-run token usage ledger.

A persistent, append-only ledger of token-usage records, indexed by
``run_id``. Each entry captures one provider call's input/output tokens,
the model name, and an optional USD cost estimate. The ledger survives
restarts so a long-running workflow's spend can be audited after the
fact.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from soteria_loop.exceptions import SoteriaError
from soteria_loop.models import TokenUsage

LedgerError = SoteriaError

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    model TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_run_id ON ledger_entries(run_id);
"""


@dataclass(frozen=True)
class LedgerEntry:
    """One persisted usage record."""

    id: int
    run_id: str
    step: int
    model: str | None
    usage: TokenUsage
    cost_usd: Decimal | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step": self.step,
            "model": self.model,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "cost_usd": str(self.cost_usd) if self.cost_usd is not None else None,
            "created_at": self.created_at.isoformat(),
        }


class TokenLedger:
    """SQLite-backed usage ledger."""

    __slots__ = ("_closed", "_connection", "_path")

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._path, isolation_level=None)
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(SCHEMA)
        except sqlite3.Error as exc:
            raise LedgerError(f"could not open ledger at {self._path}: {exc}") from exc
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._connection.close()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise LedgerError(f"ledger at {self._path} is closed")

    def record(
        self,
        run_id: str,
        *,
        step: int,
        usage: TokenUsage,
        model: str | None = None,
        cost_usd: Decimal | None = None,
    ) -> LedgerEntry:
        self._ensure_open()
        created_at = datetime.now(UTC)
        try:
            cursor = self._connection.execute(
                "INSERT INTO ledger_entries ("
                "run_id, step, model, input_tokens, output_tokens, total_tokens, "
                "cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    step,
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    str(cost_usd) if cost_usd is not None else None,
                    created_at.isoformat(),
                ),
            )
        except sqlite3.Error as exc:
            raise LedgerError(f"failed to record usage: {exc}") from exc
        return LedgerEntry(
            id=int(cursor.lastrowid or 0),
            run_id=run_id,
            step=step,
            model=model,
            usage=usage,
            cost_usd=cost_usd,
            created_at=created_at,
        )

    def entries(self, run_id: str) -> tuple[LedgerEntry, ...]:
        self._ensure_open()
        try:
            rows = self._connection.execute(
                "SELECT id, run_id, step, model, input_tokens, output_tokens, "
                "total_tokens, cost_usd, created_at FROM ledger_entries "
                "WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise LedgerError(f"failed to read entries: {exc}") from exc
        return tuple(self._row_to_entry(row) for row in rows)

    def total(self, run_id: str) -> TokenUsage:
        running = TokenUsage()
        for entry in self.entries(run_id):
            running = running.plus(entry.usage)
        return running

    def cost_total(self, run_id: str) -> Decimal | None:
        running = Decimal("0")
        seen = False
        for entry in self.entries(run_id):
            if entry.cost_usd is None:
                continue
            running += entry.cost_usd
            seen = True
        return running if seen else None

    def by_model(self, run_id: str) -> dict[str | None, TokenUsage]:
        out: dict[str | None, TokenUsage] = {}
        for entry in self.entries(run_id):
            out[entry.model] = out.get(entry.model, TokenUsage()).plus(entry.usage)
        return out

    def truncate(self, run_id: str) -> int:
        self._ensure_open()
        try:
            cursor = self._connection.execute(
                "DELETE FROM ledger_entries WHERE run_id = ?",
                (run_id,),
            )
        except sqlite3.Error as exc:
            raise LedgerError(f"failed to truncate ledger: {exc}") from exc
        return int(cursor.rowcount)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
        cost_raw = row["cost_usd"]
        cost: Decimal | None = None
        if cost_raw is not None:
            try:
                cost = Decimal(cost_raw)
            except Exception:
                cost = None
        return LedgerEntry(
            id=row["id"],
            run_id=row["run_id"],
            step=row["step"],
            model=row["model"],
            usage=TokenUsage(
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
            ),
            cost_usd=cost,
            created_at=datetime.fromisoformat(row["created_at"]),
        )


__all__ = [
    "LedgerEntry",
    "LedgerError",
    "TokenLedger",
]
