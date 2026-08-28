"""Tests for the ``git_status`` tool."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from soteria_loop.app_tools.edit_file import EditFileError
from soteria_loop.app_tools.file_tools import bind_workspace
from soteria_loop.app_tools.git_status import GitStatusArguments, git_status_tool
from soteria_loop.app_tools.workspace import Workspace


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_LFS_SKIP_SMUDGE", "1")
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


@pytest.fixture
def workspace_with_git(tmp_path: Path, isolated_home: Path) -> Workspace:
    repo = tmp_path / "ws"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("a", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "a.txt").write_text("a modified", encoding="utf-8")
    (repo / "b.txt").write_text("b", encoding="utf-8")
    return Workspace(repo, create=False)


async def _invoke(args: GitStatusArguments, workspace: Workspace) -> dict[str, object]:
    tool = git_status_tool()
    with bind_workspace(workspace):
        return await tool._function(args)  # type: ignore[no-any-return]


async def test_git_status_reports_modified_files(workspace_with_git: Workspace) -> None:
    result = await _invoke(GitStatusArguments(), workspace_with_git)
    assert result["branch"] == "main"
    assert result["clean"] is False
    assert result["modified"] == ["a.txt"]
    assert result["untracked"] == ["b.txt"]
    assert result["modified_count"] == 1
    assert result["untracked_count"] == 1


async def test_git_status_hides_untracked_when_disabled(
    workspace_with_git: Workspace,
) -> None:
    result = await _invoke(GitStatusArguments(include_untracked=False), workspace_with_git)
    assert "untracked" not in result
    assert result["untracked_count"] == 1


async def test_git_status_requires_workspace(tmp_path: Path) -> None:
    tool = git_status_tool()
    with pytest.raises(EditFileError, match="without an active workspace"):
        await tool._function(GitStatusArguments())  # type: ignore[no-any-return]


async def test_git_status_raises_for_non_git_workspace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    tool = git_status_tool()
    with bind_workspace(workspace), pytest.raises(Exception, match="not a git repository"):
        await tool._function(GitStatusArguments())  # type: ignore[no-any-return]
