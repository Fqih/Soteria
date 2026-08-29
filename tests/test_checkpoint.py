"""Tests for the checkpoint store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from soteria_loop.checkpoint import CheckpointError, CheckpointStore


def test_save_returns_first_checkpoint_with_sequence_one(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    cp = store.save("run-1", step=0, state={"phase": "init"})
    assert cp.run_id == "run-1"
    assert cp.sequence == 1
    assert cp.step == 0
    assert cp.state == {"phase": "init"}
    store.close()


def test_subsequent_saves_increment_sequence(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    a = store.save("run-1", step=1, state={"phase": "a"})
    b = store.save("run-1", step=2, state={"phase": "b"})
    c = store.save("run-1", step=3, state={"phase": "c"})
    assert a.sequence == 1
    assert b.sequence == 2
    assert c.sequence == 3
    store.close()


def test_latest_returns_highest_sequence(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"phase": "a"})
    store.save("run-1", step=2, state={"phase": "b"})
    latest = store.latest("run-1")
    assert latest is not None
    assert latest.sequence == 2
    assert latest.step == 2
    assert latest.state == {"phase": "b"}
    store.close()


def test_latest_returns_none_for_missing_run(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    assert store.latest("run-x") is None
    store.close()


def test_history_returns_all_in_order(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"k": 1})
    store.save("run-1", step=2, state={"k": 2})
    store.save("run-2", step=1, state={"k": 99})
    history = store.history("run-1")
    assert len(history) == 2
    assert [c.step for c in history] == [1, 2]
    assert [c.state for c in history] == [{"k": 1}, {"k": 2}]
    store.close()


def test_history_filters_by_run_id(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"k": 1})
    store.save("run-2", step=1, state={"k": 2})
    assert len(store.history("run-1")) == 1
    assert len(store.history("run-2")) == 1
    store.close()


def test_truncate_removes_only_target_run(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.save("run-1", step=1, state={"k": 1})
    store.save("run-2", step=1, state={"k": 2})
    deleted = store.truncate("run-1")
    assert deleted == 1
    assert store.history("run-1") == ()
    assert len(store.history("run-2")) == 1
    store.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.close()
    store.close()
    assert store._closed is True  # type: ignore[attr-defined]


def test_save_after_close_raises(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.close()
    with pytest.raises(CheckpointError, match="closed"):
        store.save("run-1", step=1, state={})


def test_latest_after_close_raises(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.close()
    with pytest.raises(CheckpointError, match="closed"):
        store.latest("run-1")


def test_history_after_close_raises(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    store.close()
    with pytest.raises(CheckpointError, match="closed"):
        store.history("run-1")


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "ckpt.db"
    store = CheckpointStore(nested)
    store.save("run-1", step=0, state={})
    store.close()
    assert nested.exists()


def test_corrupted_state_falls_back_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.db"
    store = CheckpointStore(path)
    store.save("run-1", step=1, state={"k": 1})
    store.close()
    # inject bad JSON
    conn = sqlite3.connect(path)
    conn.execute("UPDATE checkpoints SET state_json = ? WHERE run_id = 'run-1'", ("not-json",))
    conn.commit()
    conn.close()
    store2 = CheckpointStore(path)
    cp = store2.latest("run-1")
    assert cp is not None
    assert cp.state == {}
    store2.close()


def test_checkpoint_to_dict_includes_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ckpt.db")
    cp = store.save("run-1", step=5, state={"phase": "test"})
    out = cp.to_dict()
    assert out["run_id"] == "run-1"
    assert out["step"] == 5
    assert out["state"] == {"phase": "test"}
    store.close()
