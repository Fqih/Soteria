"""Tests for ``avo init`` scaffold."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest


def test_init_creates_skill_and_agents(tmp_path: Path) -> None:
    from avo.cli_init import main as init_main

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = init_main(["--cwd", str(tmp_path)])
    assert code == 0
    skill_path = tmp_path / ".avo" / "skills" / "repo-overview" / "SKILL.md"
    agents_path = tmp_path / "AGENTS.md"
    assert skill_path.is_file()
    assert agents_path.is_file()
    assert "name: repo-overview" in skill_path.read_text(encoding="utf-8")
    assert "AGENTS" in agents_path.read_text(encoding="utf-8")
    out = stdout.getvalue()
    assert "Created:" in out
    assert "python" in out  # repo kind


def test_init_is_idempotent(tmp_path: Path) -> None:
    from avo.cli_init import main as init_main

    (tmp_path / "AGENTS.md").write_text("existing", encoding="utf-8")
    skill_path = tmp_path / ".avo" / "skills" / "repo-overview" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("existing-skill", encoding="utf-8")

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = init_main(["--cwd", str(tmp_path)])
    assert code == 0
    # Existing files untouched.
    assert skill_path.read_text(encoding="utf-8") == "existing-skill"
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "existing"
    assert "No new files" in stdout.getvalue()


def test_init_detects_repo_kind(tmp_path: Path) -> None:
    from avo.cli_init import _repo_kind

    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    assert _repo_kind(tmp_path) == "rust"
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert _repo_kind(tmp_path) == "node"
    # Empty dir, no marker: generic.
    bare = tmp_path / "empty"
    bare.mkdir()
    assert _repo_kind(bare) == "generic"


def test_init_rejects_non_directory(tmp_path: Path) -> None:
    from avo.cli_init import InitCliError
    from avo.cli_init import main as init_main

    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(InitCliError):
        init_main(["--cwd", str(not_a_dir)])
