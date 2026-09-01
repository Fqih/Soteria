"""Storage protocol shared by in-memory and SQLite event stores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from avo.events import AgentEvent
from avo.models import Checkpoint, RunRecord


@runtime_checkable
class EventStore(Protocol):
    """Async interface for append-only events and mutable run metadata."""

    async def create_run(self, run: RunRecord, event: AgentEvent) -> AgentEvent:
        """Atomically create a run and its first event."""

    async def append_event(self, event: AgentEvent) -> AgentEvent:
        """Append one event and allocate its per-run sequence."""

    async def append_event_and_update_run(
        self,
        event: AgentEvent,
        run: RunRecord,
    ) -> AgentEvent:
        """Atomically append an event and update run metadata."""

    async def save_checkpoint(
        self,
        checkpoint: Checkpoint,
        event: AgentEvent,
    ) -> tuple[Checkpoint, AgentEvent]:
        """Atomically append a checkpoint event and persist its snapshot."""

    async def finalize_run(
        self,
        run: RunRecord,
        state_event: AgentEvent,
        terminal_event: AgentEvent,
    ) -> tuple[AgentEvent, AgentEvent]:
        """Atomically persist the final transition, terminal event, and metadata."""

    async def get_run(self, run_id: str) -> RunRecord:
        """Return run metadata or raise RunNotFoundError."""

    async def update_run(self, run: RunRecord) -> None:
        """Update metadata without changing the run state."""

    async def list_runs(self) -> list[RunRecord]:
        """Return all runs ordered newest first."""

    async def get_events(self, run_id: str) -> list[AgentEvent]:
        """Return events in ascending sequence order."""

    async def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Return the latest checkpoint, if one exists."""

    async def is_terminal_run(self, run_id: str) -> bool:
        """Return whether the stored run is terminal."""

    async def close(self) -> None:
        """Release resources held by the store."""
