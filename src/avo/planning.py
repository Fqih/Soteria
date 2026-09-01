"""Planning layer — ``Task`` / ``SubTask`` / ``ExecutionPlan`` / ``TaskGraph``.

The model announces its intent before executing by calling
``plan_tasks``; the runtime stores the plan in the active run's user
state so a checkpoint can resume from the same shape. The graph is a
plain DAG — there is no scheduler here; the runtime walks it as the
model declares completions.

Validation rules:

* task ids are slug-safe (``[a-z0-9][a-z0-9_-]{0,63}``)
* dependency ids must reference existing tasks
* the dependency graph must be acyclic
* a task with ``parallel_ok=True`` may run alongside its siblings
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from avo.exceptions import ToolExecutionError

PlanError = ToolExecutionError

_TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class Task(BaseModel):
    """One atomic unit of work."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4_000)
    parallel_ok: bool = Field(default=False)


class SubTask(BaseModel):
    """A leaf-level check inside a :class:`Task`."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    done: bool = Field(default=False)


class TaskGraph(BaseModel):
    """A DAG of :class:`Task` objects with dependency edges.

    ``edges`` is a mapping ``task_id -> tuple(dependency_ids, ...)``. A
    task may run once every dependency has been completed.
    """

    model_config = ConfigDict(frozen=True)

    nodes: tuple[Task, ...] = Field(default_factory=tuple)
    edges: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self) -> TaskGraph:
        ids = {task.id for task in self.nodes}
        for task in self.nodes:
            if not _TASK_ID_PATTERN.match(task.id):
                raise ValueError(f"invalid task id: {task.id!r}")
        for task_id, deps in self.edges.items():
            if task_id not in ids:
                raise ValueError(f"edge references unknown task: {task_id}")
            for dep in deps:
                if dep not in ids:
                    raise ValueError(f"task {task_id!r} depends on unknown task {dep!r}")
                if dep == task_id:
                    raise ValueError(f"task {task_id!r} cannot depend on itself")
        if _has_cycle(self.nodes, self.edges):
            raise ValueError("task graph contains a cycle")
        return self

    def ready(self, completed: Iterable[str]) -> tuple[Task, ...]:
        """Return tasks whose dependencies are all in ``completed``."""

        completed_set = frozenset(completed)
        return tuple(
            task
            for task in self.nodes
            if task.id not in completed_set
            and all(dep in completed_set for dep in self.edges.get(task.id, ()))
        )


class ExecutionPlan(BaseModel):
    """A complete plan: tasks, their order, and current progress."""

    model_config = ConfigDict(frozen=True)

    goal: str = Field(min_length=1, max_length=2_000)
    graph: TaskGraph = Field(default_factory=TaskGraph)
    subtasks: tuple[SubTask, ...] = Field(default_factory=tuple)
    completed: tuple[str, ...] = Field(default_factory=tuple)

    def with_completion(self, task_id: str) -> ExecutionPlan:
        """Return a new plan with ``task_id`` marked done."""

        if task_id not in {t.id for t in self.graph.nodes}:
            raise PlanError(f"unknown task: {task_id}")
        if task_id in self.completed:
            return self
        new_completed = (*self.completed, task_id)
        return self.model_copy(update={"completed": new_completed})

    def ready(self) -> tuple[Task, ...]:
        return self.graph.ready(self.completed)


def _has_cycle(nodes: tuple[Task, ...], edges: dict[str, tuple[str, ...]]) -> bool:
    """Return True iff the directed graph contains any cycle."""

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {task.id: WHITE for task in nodes}

    def visit(node: str) -> bool:
        if color[node] == GRAY:
            return True
        if color[node] == BLACK:
            return False
        color[node] = GRAY
        for dep in edges.get(node, ()):  # pragma: no cover - guarded
            if dep in color and visit(dep):
                return True
        color[node] = BLACK
        return False

    return any(color[task.id] == WHITE and visit(task.id) for task in nodes)


__all__ = [
    "ExecutionPlan",
    "PlanError",
    "SubTask",
    "Task",
    "TaskGraph",
]
