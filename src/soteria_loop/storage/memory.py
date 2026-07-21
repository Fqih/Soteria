"""In-memory event store used by tests and ephemeral runs."""

from __future__ import annotations

import asyncio

from soteria_loop.events import AgentEvent, EventType, validate_event_append
from soteria_loop.exceptions import RunNotFoundError, StorageError
from soteria_loop.models import Checkpoint, RunRecord
from soteria_loop.state import is_terminal


class InMemoryEventStore:
    """Append-only event storage with atomic operations under an asyncio lock."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[AgentEvent]] = {}
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("In-memory event store is closed.")

    def _require_run(self, run_id: str) -> RunRecord:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RunNotFoundError(f"Run {run_id!r} does not exist.") from exc

    def _persist_event(self, event: AgentEvent) -> AgentEvent:
        existing = self._events[event.run_id]
        validate_event_append(existing, event)
        persisted = event.model_copy(update={"sequence": len(existing) + 1}, deep=True)
        existing.append(persisted)
        return persisted.model_copy(deep=True)

    async def create_run(self, run: RunRecord, event: AgentEvent) -> AgentEvent:
        """Atomically create a run and its required RUN_CREATED event."""

        async with self._lock:
            self._ensure_open()
            if run.run_id in self._runs:
                raise StorageError(f"Run {run.run_id!r} already exists.")
            if event.run_id != run.run_id:
                raise StorageError("Run and creation event use different run IDs.")
            if event.event_type is not EventType.RUN_CREATED:
                raise StorageError("create_run requires a RUN_CREATED event.")
            self._events[run.run_id] = []
            try:
                persisted = self._persist_event(event)
            except Exception:
                del self._events[run.run_id]
                raise
            self._runs[run.run_id] = run.model_copy(deep=True)
            self._checkpoints[run.run_id] = []
            return persisted

    async def append_event(self, event: AgentEvent) -> AgentEvent:
        """Append an event with the next gap-free sequence."""

        async with self._lock:
            self._ensure_open()
            self._require_run(event.run_id)
            return self._persist_event(event)

    async def append_event_and_update_run(
        self,
        event: AgentEvent,
        run: RunRecord,
    ) -> AgentEvent:
        """Atomically append an event and replace run metadata."""

        async with self._lock:
            self._ensure_open()
            current = self._require_run(run.run_id)
            if event.run_id != run.run_id:
                raise StorageError("Run and event use different run IDs.")
            if event.event_type is EventType.STATE_CHANGED:
                if event.payload.get("from_state") != current.state.value:
                    raise StorageError("STATE_CHANGED from_state does not match stored run state.")
                if event.payload.get("to_state") != run.state.value:
                    raise StorageError("STATE_CHANGED to_state does not match updated run state.")
            persisted = self._persist_event(event)
            self._runs[run.run_id] = run.model_copy(deep=True)
            return persisted

    async def save_checkpoint(
        self,
        checkpoint: Checkpoint,
        event: AgentEvent,
    ) -> tuple[Checkpoint, AgentEvent]:
        """Atomically append CHECKPOINT_CREATED and save the matching snapshot."""

        async with self._lock:
            self._ensure_open()
            self._require_run(checkpoint.run_id)
            if event.run_id != checkpoint.run_id:
                raise StorageError("Checkpoint and event use different run IDs.")
            if event.event_type is not EventType.CHECKPOINT_CREATED:
                raise StorageError("save_checkpoint requires a CHECKPOINT_CREATED event.")
            persisted_event = self._persist_event(event)
            persisted_checkpoint = checkpoint.model_copy(
                update={"last_event_sequence": persisted_event.sequence},
                deep=True,
            )
            self._checkpoints[checkpoint.run_id].append(persisted_checkpoint)
            return (
                persisted_checkpoint.model_copy(deep=True),
                persisted_event,
            )

    async def finalize_run(
        self,
        run: RunRecord,
        state_event: AgentEvent,
        terminal_event: AgentEvent,
    ) -> tuple[AgentEvent, AgentEvent]:
        """Atomically persist the final state transition and terminal event."""

        async with self._lock:
            self._ensure_open()
            current = self._require_run(run.run_id)
            if not is_terminal(run.state):
                raise StorageError("finalize_run requires terminal run metadata.")
            if is_terminal(current.state):
                raise StorageError(f"Run {run.run_id!r} is already terminal.")
            if state_event.event_type is not EventType.STATE_CHANGED:
                raise StorageError("finalize_run requires a STATE_CHANGED event.")
            if state_event.payload.get("from_state") != current.state.value:
                raise StorageError("Final transition does not start from the stored state.")
            if state_event.payload.get("to_state") != run.state.value:
                raise StorageError("Final transition does not end at the terminal run state.")
            if terminal_event.run_id != run.run_id or state_event.run_id != run.run_id:
                raise StorageError("Final events and run metadata use different run IDs.")

            persisted_state = self._persist_event(state_event)
            try:
                persisted_terminal = self._persist_event(terminal_event)
            except Exception:
                self._events[run.run_id].pop()
                raise
            self._runs[run.run_id] = run.model_copy(deep=True)
            return persisted_state, persisted_terminal

    async def get_run(self, run_id: str) -> RunRecord:
        """Return an isolated copy of stored run metadata."""

        async with self._lock:
            self._ensure_open()
            return self._require_run(run_id).model_copy(deep=True)

    async def update_run(self, run: RunRecord) -> None:
        """Update metadata while preserving the current state."""

        async with self._lock:
            self._ensure_open()
            current = self._require_run(run.run_id)
            if run.state is not current.state:
                raise StorageError(
                    "update_run cannot change state; append a STATE_CHANGED event atomically."
                )
            self._runs[run.run_id] = run.model_copy(deep=True)

    async def list_runs(self) -> list[RunRecord]:
        """Return runs ordered by descending creation time."""

        async with self._lock:
            self._ensure_open()
            runs = sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)
            return [run.model_copy(deep=True) for run in runs]

    async def get_events(self, run_id: str) -> list[AgentEvent]:
        """Return isolated event copies in sequence order."""

        async with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            return [event.model_copy(deep=True) for event in self._events[run_id]]

    async def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Return an isolated copy of the most recent checkpoint."""

        async with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            checkpoints = self._checkpoints[run_id]
            if not checkpoints:
                return None
            return checkpoints[-1].model_copy(deep=True)

    async def is_terminal_run(self, run_id: str) -> bool:
        """Return whether a run is terminal."""

        async with self._lock:
            self._ensure_open()
            return is_terminal(self._require_run(run_id).state)

    async def close(self) -> None:
        """Mark the store closed."""

        async with self._lock:
            self._closed = True

    async def __aenter__(self) -> InMemoryEventStore:
        """Enter an async context manager."""

        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the store when leaving an async context manager."""

        await self.close()
