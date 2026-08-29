"""Run checkpointing — periodic snapshots to SQLite for crash recovery.

A checkpoint captures the full run state at a point in time so a crashed
run can resume from the last snapshot. The store is append-only — each
snapshot is keyed by ``(run_id, sequence)`` so resume picks the latest.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from hernness.exceptions import HernnessError

CheckpointError = HernnessError

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    step INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_checkpoint_run_sequence
    ON checkpoints(run_id, sequence DESC);
"""


@dataclass(frozen=True)
class Checkpoint:
    """One snapshot of a run."""

    run_id: str
    sequence: int
    step: int
    state: dict[str, JsonValue]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "step": self.step,
            "state": dict(self.state),
            "created_at": self.created_at.isoformat(),
        }


class CheckpointStore:
    """SQLite-backed checkpoint log."""

    __slots__ = ("_closed", "_connection", "_path")

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._path, isolation_level=None)
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(SCHEMA)
        except sqlite3.Error as exc:
            raise CheckpointError(
                f"could not open checkpoint store at {self._path}: {exc}"
            ) from exc
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
            raise CheckpointError(f"checkpoint store at {self._path} is closed")

    def save(
        self,
        run_id: str,
        *,
        step: int,
        state: Mapping[str, JsonValue],
    ) -> Checkpoint:
        """Append a new checkpoint. Returns the persisted record."""

        self._ensure_open()
        next_seq = self._next_sequence(run_id)
        checkpoint = Checkpoint(
            run_id=run_id,
            sequence=next_seq,
            step=step,
            state=dict(state),
            created_at=datetime.now(UTC),
        )
        try:
            self._connection.execute(
                "INSERT INTO checkpoints (run_id, sequence, step, state_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    checkpoint.run_id,
                    checkpoint.sequence,
                    checkpoint.step,
                    json.dumps(checkpoint.state),
                    checkpoint.created_at.isoformat(),
                ),
            )
        except sqlite3.Error as exc:
            raise CheckpointError(f"failed to save checkpoint: {exc}") from exc
        return checkpoint

    def latest(self, run_id: str) -> Checkpoint | None:
        """Return the highest-sequence checkpoint for ``run_id``, or ``None``."""

        self._ensure_open()
        try:
            row = self._connection.execute(
                "SELECT run_id, sequence, step, state_json, created_at FROM checkpoints "
                "WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise CheckpointError(f"failed to read latest checkpoint: {exc}") from exc
        return self._row_to_checkpoint(row) if row is not None else None

    def history(self, run_id: str) -> tuple[Checkpoint, ...]:
        """Return every checkpoint for ``run_id``, oldest first."""

        self._ensure_open()
        try:
            rows = self._connection.execute(
                "SELECT run_id, sequence, step, state_json, created_at FROM checkpoints "
                "WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise CheckpointError(f"failed to read history: {exc}") from exc
        return tuple(self._row_to_checkpoint(row) for row in rows)

    def truncate(self, run_id: str) -> int:
        """Remove every checkpoint for ``run_id``. Returns the row count deleted."""

        self._ensure_open()
        try:
            cursor = self._connection.execute(
                "DELETE FROM checkpoints WHERE run_id = ?",
                (run_id,),
            )
        except sqlite3.Error as exc:
            raise CheckpointError(f"failed to truncate checkpoints: {exc}") from exc
        return int(cursor.rowcount)

    def _next_sequence(self, run_id: str) -> int:
        try:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS max_seq FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise CheckpointError(f"failed to read sequence: {exc}") from exc
        return int(row["max_seq"]) + 1

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
        try:
            state: dict[str, JsonValue] = json.loads(row["state_json"])
        except json.JSONDecodeError:
            state = {}
        return Checkpoint(
            run_id=row["run_id"],
            sequence=row["sequence"],
            step=row["step"],
            state=state,
            created_at=datetime.fromisoformat(row["created_at"]),
        )


__all__ = [
    "Checkpoint",
    "CheckpointError",
    "CheckpointStore",
]
