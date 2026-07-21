"""Resume a durable run without duplicating a completed side effect."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from soteria import AgentEvent, AgentRuntime, EventType, FunctionTool, ModelResponse, ToolCall
from soteria.providers import FakeProvider
from soteria.storage import SQLiteEventStore


class AbruptInterruption(BaseException):
    """Represent process loss before runtime cleanup can run."""


class InterruptAfterToolResult(AgentRuntime):
    """Inject one interruption after a successful tool result is durable."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._interrupted = False

    async def _event_persisted(self, event: AgentEvent) -> None:
        if not self._interrupted and event.event_type is EventType.TOOL_COMPLETED:
            self._interrupted = True
            raise AbruptInterruption


class ChargeArguments(BaseModel):
    """Arguments for a simulated external charge."""

    amount: int


async def main() -> None:
    """Interrupt, reopen SQLite, resume, and print the recovered trace."""

    side_effects = [0]

    async def charge(arguments: ChargeArguments) -> object:
        side_effects[0] += 1
        return {"amount": arguments.amount, "charge_number": side_effects[0]}

    tool = FunctionTool(
        name="charge",
        description="Simulate a non-idempotent external charge.",
        arguments_model=ChargeArguments,
        function=charge,
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="charge-1",
                    name="charge",
                    arguments={"amount": 25},
                )
            ),
            ModelResponse(content="Charge recorded once."),
        ]
    )

    with TemporaryDirectory(prefix="soteria-example-") as directory:
        database = Path(directory) / "runs.db"
        first_store = SQLiteEventStore(database)
        first_runtime = InterruptAfterToolResult(
            provider=provider,
            tools=[tool],
            event_store=first_store,
        )
        try:
            await first_runtime.run("Charge 25 once.", run_id="resume-example")
        except AbruptInterruption:
            print("Injected interruption after the durable tool result.")
        await first_store.close()

        reopened_store = SQLiteEventStore(database)
        resumed_runtime = AgentRuntime(
            provider=FakeProvider([]),
            tools=[tool],
            event_store=reopened_store,
        )
        result = await resumed_runtime.resume("resume-example")
        trace = await resumed_runtime.inspect(result.run_id)

        print(f"Final state: {result.status.value}")
        print(f"Stop reason: {result.stop_reason.value}")
        print(f"Side effects completed: {side_effects[0]}")
        print(trace.to_text())
        await reopened_store.close()


if __name__ == "__main__":
    asyncio.run(main())
