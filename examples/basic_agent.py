"""One deterministic tool call followed by a final response."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from soteria import AgentRuntime, FunctionTool, ModelResponse, TokenUsage, ToolCall
from soteria.providers import FakeProvider


class AddArguments(BaseModel):
    """Arguments accepted by the addition tool."""

    left: int
    right: int


async def add(arguments: AddArguments) -> object:
    """Add two integers without external services."""

    return {"sum": arguments.left + arguments.right}


async def main() -> None:
    """Run and print a basic deterministic trace."""

    provider = FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="addition-1",
                    name="add",
                    arguments={"left": 2, "right": 3},
                ),
                usage=TokenUsage(input_tokens=12, output_tokens=5),
            ),
            ModelResponse(
                content="The sum is 5.",
                usage=TokenUsage(input_tokens=18, output_tokens=6),
            ),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tools=[
            FunctionTool(
                name="add",
                description="Add two integers.",
                arguments_model=AddArguments,
                function=add,
            )
        ],
    )
    result = await runtime.run("Add 2 and 3.")
    trace = await runtime.inspect(result.run_id)

    print(f"Final state: {result.status.value}")
    print(f"Stop reason: {result.stop_reason.value}")
    print(trace.to_text())


if __name__ == "__main__":
    asyncio.run(main())
