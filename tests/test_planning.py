"""Tests for the planning layer."""

from __future__ import annotations

import pytest

from soteria_loop.planning import (
    ExecutionPlan,
    PlanError,
    SubTask,
    Task,
    TaskGraph,
)


def test_task_graph_accepts_independent_nodes() -> None:
    g = TaskGraph(
        nodes=(
            Task(id="a", title="A"),
            Task(id="b", title="B"),
        )
    )
    assert {t.id for t in g.nodes} == {"a", "b"}


def test_task_graph_rejects_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        TaskGraph(
            nodes=(Task(id="a", title="A"),),
            edges={"a": ("ghost",)},
        )


def test_task_graph_rejects_self_dependency() -> None:
    with pytest.raises(ValueError, match="cannot depend on itself"):
        TaskGraph(
            nodes=(Task(id="a", title="A"),),
            edges={"a": ("a",)},
        )


def test_task_graph_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph(
            nodes=(
                Task(id="a", title="A"),
                Task(id="b", title="B"),
            ),
            edges={"a": ("b",), "b": ("a",)},
        )


def test_ready_returns_root_first() -> None:
    g = TaskGraph(
        nodes=(
            Task(id="root", title="Root"),
            Task(id="child", title="Child"),
        ),
        edges={"child": ("root",)},
    )
    assert [t.id for t in g.ready([])] == ["root"]
    assert [t.id for t in g.ready(("root",))] == ["child"]


def test_execution_plan_with_completion_is_pure() -> None:
    plan = ExecutionPlan(
        goal="ship",
        graph=TaskGraph(
            nodes=(Task(id="a", title="A"), Task(id="b", title="B")),
            edges={"b": ("a",)},
        ),
    )
    next_plan = plan.with_completion("a")
    assert plan.completed == ()
    assert next_plan.completed == ("a",)


def test_execution_plan_rejects_unknown_completion() -> None:
    plan = ExecutionPlan(
        goal="ship",
        graph=TaskGraph(nodes=(Task(id="a", title="A"),)),
    )
    with pytest.raises(PlanError, match="unknown task"):
        plan.with_completion("ghost")


def test_execution_plan_ready_after_partial_completion() -> None:
    plan = ExecutionPlan(
        goal="ship",
        graph=TaskGraph(
            nodes=(
                Task(id="a", title="A"),
                Task(id="b", title="B"),
                Task(id="c", title="C"),
            ),
            edges={"c": ("a", "b")},
        ),
    )
    completed = plan.with_completion("a").with_completion("b")
    assert [t.id for t in completed.ready()] == ["c"]


def test_execution_plan_idempotent_completion() -> None:
    plan = ExecutionPlan(
        goal="ship",
        graph=TaskGraph(nodes=(Task(id="a", title="A"),)),
    )
    once = plan.with_completion("a")
    twice = once.with_completion("a")
    assert once.completed == ("a",)
    assert twice.completed == ("a",)


def test_subtask_basic() -> None:
    sub = SubTask(task_id="a", label="first check")
    assert sub.done is False
