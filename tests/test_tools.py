"""Typed tool registry, validation, and idempotency behavior."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from avo.exceptions import (
    DuplicateToolError,
    ToolAlreadyCompletedError,
    ToolValidationError,
)
from avo.models import ToolCall
from avo.tools import FunctionTool, ToolRegistry
from tests.helpers import value_tool


@pytest.mark.asyncio
async def test_function_tool_supports_sync_callable() -> None:
    class Arguments(BaseModel):
        text: str

    tool = FunctionTool(
        name="upper",
        description="Uppercase text.",
        arguments_model=Arguments,
        function=lambda arguments: {"text": arguments.text.upper()},
    )

    assert await tool.invoke({"text": "hello"}) == {"text": "HELLO"}
    assert tool.metadata.input_schema["required"] == ["text"]


@pytest.mark.asyncio
async def test_function_tool_reports_actionable_argument_error() -> None:
    tool = value_tool()

    with pytest.raises(ToolValidationError, match=r"Invalid arguments.*value"):
        await tool.invoke({"value": "invalid"})


@pytest.mark.asyncio
async def test_function_tool_rejects_non_json_output() -> None:
    class Arguments(BaseModel):
        value: int

    tool = FunctionTool(
        name="bad-output",
        description="Return an unsupported object.",
        arguments_model=Arguments,
        function=lambda arguments: object(),
    )

    with pytest.raises(ToolValidationError, match="non-JSON"):
        await tool.invoke({"value": 1})


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(DuplicateToolError, match="unique"):
        ToolRegistry([value_tool(), value_tool()])


@pytest.mark.asyncio
async def test_registry_normalizes_missing_tool_as_failed_result() -> None:
    registry = ToolRegistry()

    result = await registry.invoke(
        ToolCall(tool_call_id="missing-call", name="missing", arguments={}),
        completed_tool_call_ids=set(),
    )

    assert result.success is False
    assert "not registered" in (result.error or "")


@pytest.mark.asyncio
async def test_registry_rejects_completed_call_identifier() -> None:
    registry = ToolRegistry([value_tool()])
    call = ToolCall(tool_call_id="already-done", name="value", arguments={"value": 1})

    with pytest.raises(ToolAlreadyCompletedError, match="already completed"):
        await registry.invoke(call, completed_tool_call_ids={"already-done"})
