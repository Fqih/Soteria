"""Show deterministic repeated-action containment before a third side effect."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from hernness import AgentRuntime, FunctionTool, LoopPolicy, ModelResponse, ToolCall
from hernness.providers import FakeProvider


class WriteArguments(BaseModel):
    """Arguments for the simulated write."""

    value: str


async def main() -> None:
    """Run a repeating script and print why Hernness stopped it."""

    writes = [0]

    async def write(arguments: WriteArguments) -> object:
        writes[0] += 1
        return {"written": arguments.value, "write_count": writes[0]}

    responses = [
        ModelResponse(
            tool_call=ToolCall(
                tool_call_id=f"write-{index}",
                name="write",
                arguments={"value": "same payload"},
            )
        )
        for index in range(1, 4)
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(responses),
        tools=[
            FunctionTool(
                name="write",
                description="Simulate a visible side effect.",
                arguments_model=WriteArguments,
                function=write,
            )
        ],
        policy=LoopPolicy(repeated_action_limit=3, no_progress_window=10),
    )
    result = await runtime.run("Keep writing the same payload.")
    trace = await runtime.inspect(result.run_id)

    print(f"Final state: {result.status.value}")
    print(f"Stop reason: {result.stop_reason.value}")
    print(f"Side effects completed: {writes[0]}")
    print(trace.to_text())


if __name__ == "__main__":
    asyncio.run(main())
