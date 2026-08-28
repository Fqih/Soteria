"""Tests for the ``plan_tasks`` tool — uses module-level runtime state."""

from __future__ import annotations

import pytest

from soteria_loop.app_tools.plan_tasks import PlanTasksArguments, plan_tasks_tool
from soteria_loop.planning import SubTask, Task


@pytest.fixture(autouse=True)
def _reset_plan_state() -> None:
    from soteria_loop.app_tools.plan_tasks import reset_active_plan

    reset_active_plan()


async def test_plan_tasks_declares_plan() -> None:
    tool = plan_tasks_tool()
    result = await tool._function(  # type: ignore[no-any-return]
        PlanTasksArguments(
            goal="ship feature",
            tasks=[Task(id="a", title="A"), Task(id="b", title="B")],
        )
    )
    assert result["goal"] == "ship feature"
    assert {t["id"] for t in result["tasks"]} == {"a", "b"}
    assert result["ready"] == ["a", "b"]


async def test_plan_tasks_marks_completion() -> None:
    tool = plan_tasks_tool()
    await tool._function(  # type: ignore[no-any-return]
        PlanTasksArguments(
            goal="ship feature",
            tasks=[Task(id="a", title="A"), Task(id="b", title="B")],
        )
    )
    result = await tool._function(PlanTasksArguments(complete="a"))  # type: ignore[no-any-return]
    assert result["completed"] == ["a"]


async def test_plan_tasks_reset_clears_state() -> None:
    tool = plan_tasks_tool()
    await tool._function(  # type: ignore[no-any-return]
        PlanTasksArguments(goal="x", tasks=[Task(id="a", title="A")])
    )
    result = await tool._function(PlanTasksArguments(reset=True, goal="y"))  # type: ignore[no-any-return]
    assert result["goal"] == "y"
    assert result["tasks"] == []


async def test_plan_tasks_no_state_returns_empty() -> None:
    tool = plan_tasks_tool()
    result = await tool._function(PlanTasksArguments())  # type: ignore[no-any-return]
    assert result["status"] == "no plan"


async def test_plan_tasks_complete_without_plan_raises() -> None:
    tool = plan_tasks_tool()
    from soteria_loop.planning import PlanError

    with pytest.raises(PlanError, match="no active plan"):
        await tool._function(PlanTasksArguments(complete="a"))


async def test_plan_tasks_with_subtasks_round_trips() -> None:
    tool = plan_tasks_tool()
    result = await tool._function(  # type: ignore[no-any-return]
        PlanTasksArguments(
            goal="ship",
            tasks=[Task(id="a", title="A")],
            subtasks=[SubTask(task_id="a", label="lint")],
        )
    )
    assert result["subtasks"] == [{"task_id": "a", "label": "lint", "done": False}]
