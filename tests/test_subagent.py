"""Tests for sub-agent delegation."""

from __future__ import annotations

from pathlib import Path

import pytest

from avo.checkpoint import CheckpointStore
from avo.subagent import (
    SubAgentError,
    SubAgentResult,
    SubAgentRunner,
    child_run_id,
)


def test_child_run_id_builds_namespaced_identifier() -> None:
    assert child_run_id("run-1", "explore") == "run-1.explore"


def test_child_run_id_rejects_empty_parent() -> None:
    with pytest.raises(SubAgentError, match="parent_run_id"):
        child_run_id("", "x")


def test_child_run_id_rejects_empty_child() -> None:
    with pytest.raises(SubAgentError, match="child_name"):
        child_run_id("run-1", "")


def test_runner_returns_result_with_child_namespace() -> None:
    def step(_run_id: str, _step: int) -> str:
        return "ok"

    runner = SubAgentRunner(run_step=step)
    result = runner.run("run-1", "explore")
    assert result.parent_run_id == "run-1"
    assert result.child_run_id == "run-1.explore"
    assert result.steps == 1
    assert result.outputs == ("ok",)


def test_runner_invokes_run_step_with_incrementing_step() -> None:
    captured: list[tuple[str, int]] = []

    def step(run_id: str, step: int) -> str:
        captured.append((run_id, step))
        return f"out-{step}"

    runner = SubAgentRunner(run_step=step)
    result = runner.run("run-1", "explore", max_steps=3)
    assert [s for _, s in captured] == [0, 1, 2]
    assert result.outputs == ("out-0", "out-1", "out-2")


def test_runner_rejects_zero_max_steps() -> None:
    runner = SubAgentRunner(run_step=lambda _r, _s: None)
    with pytest.raises(SubAgentError, match="max_steps"):
        runner.run("run-1", "explore", max_steps=0)


def test_runner_resumes_from_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    # pre-populate two child checkpoints
    store.save("run-1.explore", step=0, state={"step": 0, "phase": "init"})
    store.save("run-1.explore", step=1, state={"step": 1, "phase": "mid"})

    seen_steps: list[int] = []

    def step(_run_id: str, n: int) -> str:
        seen_steps.append(n)
        return f"out-{n}"

    runner = SubAgentRunner(run_step=step)
    result = runner.run("run-1", "explore", max_steps=2, resume_from=store)
    assert seen_steps == [2, 3]
    assert result.outputs == ("out-2", "out-3")
    store.close()


def test_runner_resumes_from_zero_when_no_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    seen: list[int] = []

    def step(_run_id: str, n: int) -> str:
        seen.append(n)
        return str(n)

    runner = SubAgentRunner(run_step=step)
    result = runner.run("run-1", "explore", max_steps=2, resume_from=store)
    assert seen == [0, 1]
    assert result.outputs == ("0", "1")
    store.close()


def test_run_many_spawns_multiple_children() -> None:
    runner = SubAgentRunner(run_step=lambda _r, _s: "ok")
    results = runner.run_many(
        "run-1",
        [
            ("explore", {}),
            ("plan", {"max_steps": 2}),
        ],
    )
    assert [r.child_run_id for r in results] == ["run-1.explore", "run-1.plan"]
    assert [r.steps for r in results] == [1, 2]


def test_subagent_result_metadata_default_empty() -> None:
    result = SubAgentResult(
        parent_run_id="run-1",
        child_run_id="run-1.x",
        steps=1,
        outputs=("ok",),
    )
    assert result.metadata == {}


def test_subagent_result_metadata_preserved() -> None:
    result = SubAgentResult(
        parent_run_id="run-1",
        child_run_id="run-1.x",
        steps=1,
        outputs=("ok",),
        metadata={"k": "v"},
    )
    assert result.metadata == {"k": "v"}
