"""Tests for snapshot/resume helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from avo.checkpoint import CheckpointStore
from avo.snapshot import ResumeError, ResumePlan, require_resume, resume


def test_resume_returns_none_for_missing_run(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    assert resume(store, "run-x") is None
    store.close()


def test_resume_returns_plan_for_existing_run(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save(
        "run-1",
        step=5,
        state={"step": 5, "phase": "planning", "extra": "data"},
    )
    plan = resume(store, "run-1")
    assert plan is not None
    assert plan.run_id == "run-1"
    assert plan.step == 5
    assert plan.phase == "planning"
    assert plan.extra_keys == ("extra",)
    store.close()


def test_resume_picks_latest_when_multiple_exist(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"step": 1, "phase": "init"})
    store.save("run-1", step=2, state={"step": 2, "phase": "start"})
    store.save("run-1", step=3, state={"step": 3, "phase": "middle"})
    plan = resume(store, "run-1")
    assert plan is not None
    assert plan.sequence == 3
    assert plan.step == 3
    assert plan.phase == "middle"
    store.close()


def test_resume_rejects_checkpoint_missing_required_keys(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"step": 1})
    with pytest.raises(ResumeError, match="missing required keys"):
        resume(store, "run-1")
    store.close()


def test_resume_rejects_wrong_step_type(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"step": "five", "phase": "x"})
    with pytest.raises(ResumeError, match="step must be int"):
        resume(store, "run-1")
    store.close()


def test_resume_rejects_wrong_phase_type(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"step": 1, "phase": 99})
    with pytest.raises(ResumeError, match="phase must be str"):
        resume(store, "run-1")
    store.close()


def test_require_resume_raises_when_missing(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    with pytest.raises(ResumeError, match="no checkpoint"):
        require_resume(store, "run-x")
    store.close()


def test_require_resume_returns_plan_when_present(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"step": 1, "phase": "init"})
    plan = require_resume(store, "run-1")
    assert plan.step == 1
    store.close()


def test_resume_plan_from_checkpoint_directly(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    cp = store.save("run-1", step=2, state={"step": 2, "phase": "active", "k": "v"})
    plan = ResumePlan.from_checkpoint(cp)
    assert plan.extra_keys == ("k",)
    assert plan.state == {"step": 2, "phase": "active", "k": "v"}
    store.close()


def test_resume_plan_preserves_extra_keys(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save(
        "run-1",
        step=1,
        state={"step": 1, "phase": "x", "a": 1, "b": 2, "c": 3},
    )
    plan = resume(store, "run-1")
    assert plan is not None
    assert sorted(plan.extra_keys) == ["a", "b", "c"]
    assert plan.state["a"] == 1
    store.close()
