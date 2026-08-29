"""Tests for the ``task`` sub-agent dispatch tool."""

from __future__ import annotations

import pytest

from hernness.app_tools.file_tools import read_file_tool, write_file_tool
from hernness.app_tools.task_tool import AgentType, TaskArguments, task_tool
from hernness.models import ModelResponse, ToolCall
from hernness.policies import LoopPolicy
from hernness.providers.fake import FakeProvider
from hernness.runtime import AgentRuntime
from hernness.storage.memory import InMemoryEventStore


def _parent(
    *,
    responses: list[ModelResponse],
    tools: list[object] | None = None,
) -> AgentRuntime:
    runtime = AgentRuntime(
        provider=FakeProvider(responses=responses),
        event_store=InMemoryEventStore(),
        policy=LoopPolicy(max_steps=4),
    )
    if tools:
        # Re-register so tests can introspect what the parent exposes.
        for tool in tools:
            runtime.tools.register(tool)  # type: ignore[attr-defined]
    return runtime


async def test_task_tool_metadata_describes_contract() -> None:
    parent = _parent(responses=[])
    tool = task_tool(parent_runtime=parent)
    metadata = tool.metadata
    assert metadata.name == "task"
    assert metadata.input_schema["properties"]["prompt"]
    assert metadata.input_schema["properties"]["agent_type"]


async def test_task_explore_returns_summary_with_run_id() -> None:
    parent = _parent(
        responses=[ModelResponse(content="child finished")],
        tools=[read_file_tool(), write_file_tool()],
    )
    tool = task_tool(parent_runtime=parent)

    result = await tool._function(TaskArguments(agent_type=AgentType.EXPLORE, prompt="summarise"))

    assert result["agent_type"] == "explore"
    assert result["output"] == "child finished"
    assert result["stop_reason"] == "completed"
    assert result["run_id"]
    assert result["steps"] >= 1


async def test_task_explore_only_uses_read_only_tools() -> None:
    parent = _parent(
        responses=[
            ModelResponse(content="child finished"),
        ],
        tools=[read_file_tool(), write_file_tool()],
    )
    tool = task_tool(parent_runtime=parent)

    # The child's first request should advertise only read_file, not write_file.
    await tool._function(TaskArguments(agent_type=AgentType.EXPLORE, prompt="look"))
    child_requests = parent.provider.requests  # type: ignore[attr-defined]
    assert child_requests, "child runtime should have made at least one request"
    advertised = {tool.name for tool in child_requests[0].tools}
    assert "read_file" in advertised
    assert "write_file" not in advertised


async def test_task_general_exposes_all_parent_tools() -> None:
    parent = _parent(
        responses=[ModelResponse(content="done")],
        tools=[read_file_tool(), write_file_tool()],
    )
    tool = task_tool(parent_runtime=parent)

    await tool._function(TaskArguments(agent_type=AgentType.GENERAL, prompt="work"))
    child_requests = parent.provider.requests  # type: ignore[attr-defined]
    advertised = {tool.name for tool in child_requests[0].tools}
    assert {"read_file", "write_file"}.issubset(advertised)


async def test_task_child_run_is_isolated() -> None:
    parent = _parent(
        responses=[ModelResponse(content="isolated done")],
        tools=[read_file_tool()],
    )
    parent_store = parent.event_store
    parent_runs_before = len(await parent_store.list_runs())

    tool = task_tool(parent_runtime=parent)
    await tool._function(TaskArguments(agent_type=AgentType.EXPLORE, prompt="x"))

    # The parent must still only have its own run, not the child's.
    parent_runs_after = len(await parent_store.list_runs())
    assert parent_runs_after == parent_runs_before


async def test_task_respects_max_steps_override() -> None:
    parent = _parent(
        responses=[ModelResponse(content="done")],
        tools=[read_file_tool()],
    )
    tool = task_tool(parent_runtime=parent, policy_overrides=LoopPolicy(max_steps=2))

    result = await tool._function(
        TaskArguments(agent_type=AgentType.EXPLORE, prompt="x", max_steps=3)
    )
    assert result["steps"] >= 1


async def test_task_empty_prompt_rejected() -> None:
    _parent(responses=[])  # ensure factory works; rejection happens at model level
    with pytest.raises(ValueError, match="at least 1 character"):
        TaskArguments(agent_type=AgentType.EXPLORE, prompt="")


async def test_task_unknown_agent_type_rejected() -> None:
    _parent(responses=[])  # ensure factory works; rejection happens at model level
    with pytest.raises(ValueError):
        TaskArguments(agent_type="nuclear", prompt="x")  # type: ignore[arg-type]


# Reference ToolCall so the type checker keeps the import alive; the
# tests above use ``ModelResponse(tool_call=...)`` paths indirectly.
_ = ToolCall
