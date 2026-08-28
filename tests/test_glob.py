"""Tests for the ``glob`` tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from soteria_loop.app_tools.edit_file import EditFileError
from soteria_loop.app_tools.file_tools import bind_workspace
from soteria_loop.app_tools.glob_tool import GlobArguments, glob_tool
from soteria_loop.app_tools.workspace import Workspace, WorkspacePathError


@pytest.fixture
def workspace_with_files(tmp_path: Path) -> Workspace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "src" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("ta", encoding="utf-8")
    (tmp_path / "README.md").write_text("r", encoding="utf-8")
    return Workspace(tmp_path, create=True)


async def _invoke(args: GlobArguments) -> dict[str, object]:
    tool = glob_tool()
    return await tool._function(args)  # type: ignore[no-any-return]


async def test_glob_recursive_matches_all_python_files(
    workspace_with_files: Workspace,
) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GlobArguments(pattern="**/*.py"))
    assert sorted(result["matches"]) == ["src/a.py", "src/b.py", "tests/test_a.py"]
    assert result["count"] == 3
    assert result["truncated"] is False


async def test_glob_non_recursive_matches_immediate_children(
    workspace_with_files: Workspace,
) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GlobArguments(pattern="*.md", recursive=False))
    assert result["matches"] == ["README.md"]


async def test_glob_truncates_at_max_results(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    for i in range(5):
        (tmp_path / f"file_{i}.txt").write_text(str(i), encoding="utf-8")
    with bind_workspace(workspace):
        result = await _invoke(GlobArguments(pattern="*.txt", recursive=False, max_results=2))
    assert len(result["matches"]) == 2
    assert result["truncated"] is True


async def test_glob_rejects_path_outside_workspace(
    workspace_with_files: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    with bind_workspace(workspace_with_files), pytest.raises(WorkspacePathError):
        await _invoke(GlobArguments(pattern="*", path=str(outside)))


async def test_glob_requires_directory_path(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    file = tmp_path / "thing.txt"
    file.write_text("x", encoding="utf-8")
    with bind_workspace(workspace), pytest.raises(EditFileError, match="not a directory"):
        await _invoke(GlobArguments(pattern="*", path=str(file)))


async def test_glob_raises_without_active_workspace(tmp_path: Path) -> None:
    with pytest.raises(EditFileError, match="without an active workspace"):
        await _invoke(GlobArguments(pattern="*.py", path=str(tmp_path)))


async def test_glob_empty_match_returns_empty_list(
    workspace_with_files: Workspace,
) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GlobArguments(pattern="**/*.nope"))
    assert result["matches"] == []
    assert result["count"] == 0
