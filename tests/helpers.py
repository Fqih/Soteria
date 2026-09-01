"""Shared deterministic helpers for Avo tests."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from avo.events import AgentEvent, EventType
from avo.models import RunRecord
from avo.storage.base import EventStore
from avo.tools import FunctionTool


class ValueArguments(BaseModel):
    """One integer tool argument."""

    value: int


def value_tool(
    *,
    name: str = "value",
    output: object | None = None,
    counter: list[int] | None = None,
    raises: Exception | None = None,
) -> FunctionTool[ValueArguments]:
    """Build a deterministic typed tool for tests."""

    async def invoke(arguments: ValueArguments) -> object:
        if counter is not None:
            counter[0] += 1
        if raises is not None:
            raise raises
        if output is not None:
            return output
        return {"value": arguments.value}

    return FunctionTool(
        name=name,
        description=f"Return a deterministic value from {name}.",
        arguments_model=ValueArguments,
        function=invoke,
    )


async def seed_run(
    store: EventStore,
    *,
    run_id: str = "run-1",
    task: str = "test task",
) -> RunRecord:
    """Create a non-terminal run with its creation event."""

    run = RunRecord(run_id=run_id, task=task)
    await store.create_run(
        run,
        AgentEvent(
            run_id=run_id,
            event_type=EventType.RUN_CREATED,
            payload={"task": task},
        ),
    )
    return run


ClockAdvancer = Callable[[], object]
