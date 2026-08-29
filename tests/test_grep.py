"""Tests for the ``grep`` tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from hernness.app_tools.edit_file import EditFileError
from hernness.app_tools.file_tools import bind_workspace
from hernness.app_tools.grep_tool import GrepArguments, grep_tool
from hernness.app_tools.workspace import Workspace, WorkspacePathError


@pytest.fixture
def workspace_with_files(tmp_path: Path) -> Workspace:
    py_content = "def hello():\n    return 42\n\n# hello world\n"
    (tmp_path / "a.py").write_text(py_content, encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text\nhello there\n", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01\x02hello\x00")
    return Workspace(tmp_path, create=True)


async def _invoke(args: GrepArguments) -> dict[str, object]:
    tool = grep_tool()
    return await tool._function(args)  # type: ignore[no-any-return]


async def test_grep_finds_matches_across_files(workspace_with_files: Workspace) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GrepArguments(pattern=r"hello"))
    by_path: dict[str, list[int]] = {}
    for m in result["matches"]:  # type: ignore[index]
        by_path.setdefault(m["path"], []).append(m["line_number"])
    assert sorted(by_path) == ["a.py", "b.txt"]
    assert sorted(by_path["a.py"]) == [1, 4]
    assert by_path["b.txt"] == [2]


async def test_grep_skips_binary_files(workspace_with_files: Workspace) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GrepArguments(pattern="hello"))
    paths = [m["path"] for m in result["matches"]]  # type: ignore[index]
    assert "c.bin" not in paths


async def test_grep_case_insensitive(workspace_with_files: Workspace) -> None:
    (workspace_with_files.root / "caps.txt").write_text("HELLO world\n", encoding="utf-8")
    with bind_workspace(workspace_with_files):
        sensitive = await _invoke(GrepArguments(pattern="hello"))
        insensitive = await _invoke(GrepArguments(pattern="hello", case_insensitive=True))
    # sensitive: 2 in a.py (lines 1+4) + 1 in b.txt = 3
    assert sensitive["count"] == 3
    # insensitive: + 1 in caps.txt = 4
    assert insensitive["count"] == 4


async def test_grep_include_glob_filter(workspace_with_files: Workspace) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GrepArguments(pattern="hello", include_glob="*.py"))
    paths = sorted({m["path"] for m in result["matches"]})  # type: ignore[index]
    assert paths == ["a.py"]


async def test_grep_context_lines(workspace_with_files: Workspace) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GrepArguments(pattern=r"return 42", context_lines=2))
    matches = result["matches"]  # type: ignore[index]
    assert len(matches) == 1
    ctx = matches[0]["context"]
    assert "def hello():" in ctx["before"]
    assert "# hello world" in " ".join(ctx["after"])


async def test_grep_truncates_at_max_results(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    lines = "\n".join(f"hit {i}" for i in range(20))
    (tmp_path / "f.txt").write_text(lines, encoding="utf-8")
    with bind_workspace(workspace):
        result = await _invoke(GrepArguments(pattern="hit", max_results=5))
    assert result["count"] == 5
    assert result["truncated"] is True


async def test_grep_rejects_invalid_regex(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    with bind_workspace(workspace), pytest.raises(EditFileError, match="invalid regex"):
        await _invoke(GrepArguments(pattern="[unclosed"))


async def test_grep_rejects_path_outside_workspace(
    workspace_with_files: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    with bind_workspace(workspace_with_files), pytest.raises(WorkspacePathError):
        await _invoke(GrepArguments(pattern="x", path=str(outside)))


async def test_grep_raises_without_workspace(tmp_path: Path) -> None:
    with pytest.raises(EditFileError, match="without an active workspace"):
        await _invoke(GrepArguments(pattern="x", path=str(tmp_path)))


async def test_grep_rejects_non_directory_path(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    f = tmp_path / "f.txt"
    f.write_text("hello", encoding="utf-8")
    with bind_workspace(workspace), pytest.raises(EditFileError, match="not a directory"):
        await _invoke(GrepArguments(pattern="hello", path=str(f)))


async def test_grep_empty_match_returns_zero(workspace_with_files: Workspace) -> None:
    with bind_workspace(workspace_with_files):
        result = await _invoke(GrepArguments(pattern=r"^nothing$"))
    assert result["count"] == 0
    assert result["matches"] == []
