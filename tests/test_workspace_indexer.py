"""Tests for the WorkspaceIndexer."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hernness.app_tools.file_tools import bind_workspace
from hernness.app_tools.workspace import Workspace
from hernness.app_tools.workspace_map import WorkspaceMapArguments, workspace_map_tool
from hernness.workspace.indexer import WorkspaceIndexer


@pytest.fixture
def workspace_with_tree(tmp_path: Path) -> Workspace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / ".hidden").write_text("h", encoding="utf-8")
    return Workspace(tmp_path, create=False)


def test_index_returns_relative_paths(workspace_with_tree: Workspace) -> None:
    index = WorkspaceIndexer(workspace_with_tree.root).build()
    rels = {entry.path for entry in index.entries}
    assert "src/a.py" in rels
    assert "src/b.py" in rels
    assert "README.md" in rels
    assert ".hidden" in rels
    # Never leaks absolute paths.
    for entry in index.entries:
        assert not entry.path.startswith("/")


def test_index_records_size_and_mtime(workspace_with_tree: Workspace) -> None:
    index = WorkspaceIndexer(workspace_with_tree.root).build()
    a = next(e for e in index.entries if e.path == "src/a.py")
    assert a.size == 1
    assert a.mtime_ns > 0


def test_index_caps_max_entries(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    index = WorkspaceIndexer(workspace.root).build(max_entries=3)
    assert len(index.entries) == 3
    assert index.truncated is True


def test_index_respects_max_depth(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "leaf.txt").write_text("x", encoding="utf-8")
    index = WorkspaceIndexer(workspace.root).build(max_depth=2)
    paths = {e.path for e in index.entries}
    assert not any("leaf.txt" in p for p in paths)


def test_recent_files_orders_by_mtime(workspace_with_tree: Workspace) -> None:
    (workspace_with_tree.root / "fresh.txt").write_text("z", encoding="utf-8")
    # Force mtime difference by sleeping briefly then updating README.
    time.sleep(0.01)
    (workspace_with_tree.root / "README.md").write_text("updated", encoding="utf-8")
    index = WorkspaceIndexer(workspace_with_tree.root).build()
    recent = index.recent_files(limit=2)
    assert len(recent) == 2
    assert recent[0].path == "README.md"  # latest edit


def test_map_text_truncates(workspace_with_tree: Workspace) -> None:
    index = WorkspaceIndexer(workspace_with_tree.root).build()
    rendered = index.map_text(max_entries=2)
    assert "+2 more" in rendered


def test_index_raises_on_missing_root(tmp_path: Path) -> None:
    bogus = tmp_path / "no-such"
    with pytest.raises(Exception, match="not a directory"):
        WorkspaceIndexer(bogus).build()


async def test_workspace_map_tool_returns_payload(workspace_with_tree: Workspace) -> None:
    tool = workspace_map_tool()
    with bind_workspace(workspace_with_tree):
        result = await tool._function(WorkspaceMapArguments())  # type: ignore[no-any-return]
    assert result["root"] == str(workspace_with_tree.root.resolve())
    assert "src/a.py" in result["map"]
    assert isinstance(result["recent"], list)


async def test_workspace_map_tool_can_skip_sections(
    workspace_with_tree: Workspace,
) -> None:
    tool = workspace_map_tool()
    with bind_workspace(workspace_with_tree):
        result = await tool._function(
            WorkspaceMapArguments(include_map=False, include_recent=False)
        )
    assert "map" not in result
    assert "recent" not in result


async def test_workspace_map_tool_requires_workspace(tmp_path: Path) -> None:
    tool = workspace_map_tool()
    from hernness.app_tools.edit_file import EditFileError

    with pytest.raises(EditFileError, match="without an active workspace"):
        await tool._function(WorkspaceMapArguments())  # type: ignore[no-any-return]
