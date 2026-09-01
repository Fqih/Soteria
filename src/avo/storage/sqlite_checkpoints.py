"""Checkpoint persistence operations for the SQLite event store.

The checkpoint methods are extracted from ``storage.sqlite`` so the
store file itself can stay under the project's ~400 line target. The
public surface (``SQLiteEventStore.save_checkpoint`` and
``SQLiteEventStore.get_latest_checkpoint``) is unchanged; the methods
simply delegate to the free functions below.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pydantic import ValidationError

from avo.events import AgentEvent, EventType
from avo.exceptions import AvoError, StorageError
from avo.models import Checkpoint

if TYPE_CHECKING:
    from avo.storage.sqlite import SQLiteEventStore


async def save_checkpoint(
    store: SQLiteEventStore,
    checkpoint: Checkpoint,
    event: AgentEvent,
) -> tuple[Checkpoint, AgentEvent]:
    """Atomically append ``CHECKPOINT_CREATED`` and persist its snapshot."""

    async with store._lock:
        store._ensure_open()
        if event.run_id != checkpoint.run_id:
            raise StorageError("Checkpoint and event use different run IDs.")
        if event.event_type is not EventType.CHECKPOINT_CREATED:
            raise StorageError("save_checkpoint requires a CHECKPOINT_CREATED event.")
        try:
            store._begin()
            store._require_run_row(checkpoint.run_id)
            persisted_event = store._insert_event(event)
            persisted_checkpoint = checkpoint.model_copy(
                update={"last_event_sequence": persisted_event.sequence},
                deep=True,
            )
            store._connection.execute(
                """
                INSERT INTO checkpoints(
                    checkpoint_id, run_id, event_sequence, created_at, checkpoint_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    persisted_checkpoint.checkpoint_id,
                    persisted_checkpoint.run_id,
                    persisted_checkpoint.last_event_sequence,
                    persisted_checkpoint.created_at.isoformat(),
                    persisted_checkpoint.model_dump_json(),
                ),
            )
            store._commit()
            return persisted_checkpoint, persisted_event
        except AvoError:
            store._rollback()
            raise
        except (sqlite3.Error, ValidationError, ValueError, TypeError) as exc:
            store._rollback()
            raise store._raise_storage_error("save a checkpoint", exc) from exc


async def get_latest_checkpoint(
    store: SQLiteEventStore,
    run_id: str,
) -> Checkpoint | None:
    """Return the latest checkpoint for a run, or ``None`` if none exists."""

    async with store._lock:
        store._ensure_open()
        try:
            store._require_run_row(run_id)
            row = store._connection.execute(
                """
                SELECT checkpoint_json
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY event_sequence DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return Checkpoint.model_validate_json(str(row["checkpoint_json"]))
        except AvoError:
            raise
        except (sqlite3.Error, ValidationError, ValueError, TypeError) as exc:
            raise store._raise_storage_error("read a checkpoint", exc) from exc


__all__ = ["get_latest_checkpoint", "save_checkpoint"]
