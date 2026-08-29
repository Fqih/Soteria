"""Tests for the Skills loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from hernness.skills import SkillError, SkillRegistry, load_skills


def _write_skill(root: Path, name: str, body: str) -> Path:
    path = root / f"{name}.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_registry_loads_known_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path, "review-pr", "Review the diff for correctness.")
    registry = SkillRegistry(tmp_path)
    assert registry.exists("review-pr")
    assert registry.load("review-pr") == "Review the diff for correctness."


def test_registry_unknown_skill_raises(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    with pytest.raises(SkillError, match="unknown skill"):
        registry.load("nope")


def test_registry_lists_names_sorted(tmp_path: Path) -> None:
    _write_skill(tmp_path, "zeta", "z")
    _write_skill(tmp_path, "alpha", "a")
    _write_skill(tmp_path, "mu", "m")
    # Hidden / non-md files must be skipped.
    (tmp_path / "not-md.txt").write_text("x", encoding="utf-8")
    (tmp_path / "README").write_text("x", encoding="utf-8")
    assert SkillRegistry(tmp_path).names() == ["alpha", "mu", "zeta"]


def test_registry_rejects_invalid_name(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    with pytest.raises(SkillError, match="invalid skill name"):
        registry.load("../escape")


def test_registry_rejects_name_with_path_separator(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    with pytest.raises(SkillError):
        registry.load("sub/dir")


def test_registry_iter_yields_pairs(tmp_path: Path) -> None:
    _write_skill(tmp_path, "a", "A body")
    _write_skill(tmp_path, "b", "B body")
    pairs = dict(SkillRegistry(tmp_path))
    assert pairs == {"a": "A body", "b": "B body"}


def test_registry_handles_missing_root(tmp_path: Path) -> None:
    bogus = tmp_path / "no-such-dir"
    registry = SkillRegistry(bogus)
    assert registry.names() == []
    assert not registry.exists("anything")


def test_load_skills_creates_missing_root(tmp_path: Path) -> None:
    root = tmp_path / "fresh-skills"
    assert not root.exists()
    skills = load_skills(root)
    assert skills == {}
    assert root.is_dir()


def test_load_skills_returns_all(tmp_path: Path) -> None:
    _write_skill(tmp_path, "x", "X body")
    _write_skill(tmp_path, "y", "Y body")
    assert load_skills(tmp_path) == {"x": "X body", "y": "Y body"}
