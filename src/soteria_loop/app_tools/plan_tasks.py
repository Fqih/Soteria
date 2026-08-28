"""``plan_tasks`` tool — declare a structured execution plan.

The runtime stores the plan in module-level state for the lifetime of
the current run. The model calls the tool once to declare intent and
again with ``complete=<task_id>`` to mark progress.

The store is intentionally simple — the runtime's checkpoint system
serialises the active plan via :func:`get_active_plan` when persisting.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from soteria_loop import FunctionTool as PublicFunctionTool
from soteria_loop.planning import ExecutionPlan, PlanError, SubTask, Task, TaskGraph

# Module-level state for the currently-active execution plan.
_ACTIVE_PLAN: ExecutionPlan | None = None


def _current_plan() -> ExecutionPlan | None:
    return _ACTIVE_PLAN


def _store_plan(plan: ExecutionPlan) -> None:
    global _ACTIVE_PLAN
    _ACTIVE_PLAN = plan


def get_active_plan() -> dict[str, Any] | None:
    """Return the serialised active plan or ``None``."""

    return _ACTIVE_PLAN.model_dump(mode="json") if _ACTIVE_PLAN is not None else None


def reset_active_plan() -> None:
    """Clear the active plan — used by tests and turn boundaries."""

    global _ACTIVE_PLAN
    _ACTIVE_PLAN = None


class PlanTasksArguments(BaseModel):
    """Arguments for the ``plan_tasks`` tool."""

    goal: str | None = Field(default=None, min_length=1, max_length=2_000)
    tasks: list[Task] | None = Field(default=None, max_length=200)
    subtasks: list[SubTask] | None = Field(default=None, max_length=500)
    complete: str | None = Field(default=None, min_length=1, max_length=64)
    reset: bool = Field(default=False)


def _serialise(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "goal": plan.goal,
        "tasks": [task.model_dump() for task in plan.graph.nodes],
        "edges": {k: list(v) for k, v in plan.graph.edges.items()},
        "subtasks": [sub.model_dump() for sub in plan.subtasks],
        "completed": list(plan.completed),
        "ready": [task.id for task in plan.ready()],
    }


async def _plan_tasks(arguments: PlanTasksArguments) -> dict[str, Any]:
    current = _current_plan()

    if arguments.reset:
        _store_plan(ExecutionPlan(goal=arguments.goal or ""))
        return _serialise(_current_plan() or ExecutionPlan(goal=""))

    if arguments.complete is not None:
        if current is None:
            raise PlanError("no active plan; declare one with plan_tasks first")
        current = current.with_completion(arguments.complete)
        _store_plan(current)
        return _serialise(current)

    if arguments.goal is None and not arguments.tasks:
        if current is None:
            return {"status": "no plan", "goal": None, "tasks": [], "completed": []}
        return _serialise(current)

    if arguments.goal is None:
        raise PlanError("plan_tasks requires goal when declaring a new plan")

    graph = TaskGraph(nodes=tuple(arguments.tasks or ()))
    plan = ExecutionPlan(
        goal=arguments.goal,
        graph=graph,
        subtasks=tuple(arguments.subtasks or ()),
    )
    _store_plan(plan)
    return _serialise(plan)


def plan_tasks_tool() -> PublicFunctionTool[PlanTasksArguments]:
    """Return a :class:`FunctionTool` that manages execution plans."""

    return PublicFunctionTool(
        name="plan_tasks",
        description=(
            "Declare or update the active execution plan. Pass ``goal`` "
            "and ``tasks`` to start a plan, ``complete=<task_id>`` to "
            "mark progress, or ``reset=True`` to clear. The plan is "
            "stored in the run state so a resumed run continues where "
            "it left off."
        ),
        arguments_model=PlanTasksArguments,
        function=_plan_tasks,
    )


__all__ = ["PlanTasksArguments", "plan_tasks_tool"]
