"""Path-traversal tests for ``soteria_loop.app_tools.workspace``.

These are the most security-sensitive tests in the runtime: a single missed
escape vector turns the agent into a path-traversal primitive. Read every
case carefully when reviewing changes here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from soteria_loop.app_tools.workspace import (
    Workspace,
    WorkspacePathError,
    validate_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_tree(tmp_path: Path) -> Path:
    """Lay out a small directory tree under ``tmp_path`` for traversal tests.

    Layout::

        tmp_path/
            file.txt
            sub/
                nested.txt
                link_to_outside -> /etc
                link_to_root -> tmp_path
    """

    root = tmp_path
    (root / "file.txt").write_text("hello", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")
    # Symlink inside root pointing outside — must be rejected.
    outside_dir = tmp_path.parent / "outside-target"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")
    os.symlink(outside_dir, sub / "link_to_outside")
    # Symlink inside root pointing back at root — allowed because resolved
    # target stays inside.
    os.symlink(root, sub / "link_to_root")
    return root


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_workspace_resolves_relative_root(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("x", encoding="utf-8")
    ws = Workspace(sub)
    result = ws.validate_path("nested.txt")
    assert result == (sub / "nested.txt").resolve()


def test_workspace_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(WorkspacePathError, match="does not exist"):
        Workspace(tmp_path / "missing")


def test_workspace_create_makes_directory(tmp_path: Path) -> None:
    new_root = tmp_path / "fresh"
    Workspace(new_root, create=True)
    assert new_root.is_dir()


def test_workspace_rejects_file_as_root(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(WorkspacePathError, match="not a directory"):
        Workspace(file_path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_validate_returns_resolved_path_inside_root(workspace_tree: Path) -> None:
    result = Workspace(workspace_tree).validate_path("file.txt")
    assert result == (workspace_tree / "file.txt").resolve()
    assert result.parent == workspace_tree.resolve()


def test_validate_accepts_nested_path(workspace_tree: Path) -> None:
    result = Workspace(workspace_tree).validate_path("sub/nested.txt")
    assert result == (workspace_tree / "sub" / "nested.txt").resolve()


def test_validate_accepts_absolute_path_inside_root(workspace_tree: Path) -> None:
    absolute = str((workspace_tree / "sub" / "nested.txt").resolve())
    result = Workspace(workspace_tree).validate_path(absolute)
    assert result == (workspace_tree / "sub" / "nested.txt").resolve()


def test_validate_allows_path_equal_to_root(workspace_tree: Path) -> None:
    """Listing the root directory is a legitimate operation."""

    result = Workspace(workspace_tree).validate_path(str(workspace_tree.resolve()))
    assert result == workspace_tree.resolve()


# ---------------------------------------------------------------------------
# Traversal rejection
# ---------------------------------------------------------------------------


def test_validate_rejects_parent_traversal(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path("../file.txt")


def test_validate_rejects_deep_parent_traversal(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path("sub/../../outside-target/secret.txt")


def test_validate_rejects_absolute_outside_path(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path("/etc/passwd")


def test_validate_rejects_symlink_inside_pointing_outside(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path("sub/link_to_outside/secret.txt")


def test_validate_rejects_symlink_to_root(workspace_tree: Path) -> None:
    """A symlink whose target is the root is technically inside, but the
    double resolution makes the path confusing. We accept it because the
    resolved target is the workspace root itself."""

    result = Workspace(workspace_tree).validate_path("sub/link_to_root/file.txt")
    assert result == (workspace_tree / "file.txt").resolve()


def test_validate_rejects_parent_dotdot_in_middle(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path("sub/../sub/../../outside-target")


def test_validate_rejects_null_byte(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="null byte"):
        Workspace(workspace_tree).validate_path("file.txt\x00.png")


def test_validate_rejects_empty_string(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="empty"):
        Workspace(workspace_tree).validate_path("")


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


def test_validate_rejects_nonexistent_path(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="does not exist"):
        Workspace(workspace_tree).validate_path("missing.txt")


def test_validate_must_exist_false_allows_new_file(workspace_tree: Path) -> None:
    result = Workspace(workspace_tree).validate_path("new.txt", must_exist=False)
    assert result == (workspace_tree / "new.txt").resolve()


def test_validate_must_exist_false_still_rejects_escape(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path(
            "../nonexistent-but-traversed.txt", must_exist=False
        )


def test_validate_must_exist_false_rejects_nonexistent_parent(
    workspace_tree: Path,
) -> None:
    with pytest.raises(WorkspacePathError, match="parent"):
        Workspace(workspace_tree).validate_path("missing-dir/new.txt", must_exist=False)


# ---------------------------------------------------------------------------
# validate_for_write
# ---------------------------------------------------------------------------


def test_validate_for_write_allows_new_file(workspace_tree: Path) -> None:
    result = Workspace(workspace_tree).validate_for_write("new.txt")
    assert result == (workspace_tree / "new.txt").resolve()


def test_validate_for_write_allows_existing_file(workspace_tree: Path) -> None:
    result = Workspace(workspace_tree).validate_for_write("file.txt")
    assert result == (workspace_tree / "file.txt").resolve()


def test_validate_for_write_rejects_parent_traversal(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_for_write("../escape.txt")


def test_validate_for_write_rejects_absolute_outside(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_for_write("/tmp/escape.txt")


def test_validate_for_write_rejects_symlink_inside_to_outside(
    workspace_tree: Path,
) -> None:
    with pytest.raises(WorkspacePathError, match="symlink"):
        Workspace(workspace_tree).validate_for_write("sub/link_to_outside/x.txt")


def test_validate_for_write_rejects_null_byte(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="null byte"):
        Workspace(workspace_tree).validate_for_write("file.txt\x00.png")


def test_validate_for_write_rejects_empty(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError, match="empty"):
        Workspace(workspace_tree).validate_for_write("")


# ---------------------------------------------------------------------------
# Symlink edge cases
# ---------------------------------------------------------------------------


def test_validate_rejects_resolve_failure(workspace_tree: Path) -> None:
    """A broken symlink must fail closed (not silently fall back)."""

    os.symlink("/nonexistent-target", workspace_tree / "sub" / "broken_link")
    with pytest.raises(WorkspacePathError):
        Workspace(workspace_tree).validate_path("sub/broken_link")


def test_validate_chain_of_symlinks_resolved(workspace_tree: Path) -> None:
    """Two symlinks chained — workspace sees the final target."""

    os.symlink(
        workspace_tree / "sub" / "link_to_outside",
        workspace_tree / "sub" / "chain_link",
    )
    with pytest.raises(WorkspacePathError, match="escapes workspace root"):
        Workspace(workspace_tree).validate_path("sub/chain_link/secret.txt")


def test_validate_symlink_to_file_inside(workspace_tree: Path) -> None:
    os.symlink(workspace_tree / "file.txt", workspace_tree / "sub" / "file_link")
    result = Workspace(workspace_tree).validate_path("sub/file_link")
    assert result == (workspace_tree / "file.txt").resolve()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def test_module_level_validate_path(workspace_tree: Path) -> None:
    result = validate_path(workspace_tree, "file.txt")
    assert result == (workspace_tree / "file.txt").resolve()


def test_module_level_validate_path_rejects_escape(workspace_tree: Path) -> None:
    with pytest.raises(WorkspacePathError):
        validate_path(workspace_tree, "../escape.txt")
