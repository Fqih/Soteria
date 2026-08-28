"""``git_status`` tool — surface repository state to the model.

The tool binds to the active :class:`Workspace` and walks up to the
nearest git working tree. Path traversal is rejected at construction;
the tool only ever returns paths inside the workspace root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from soteria_loop import FunctionTool as PublicFunctionTool

from ..workspace.git import GitRepository, GitStatus
from .edit_file import EditFileError
from .file_tools import _workspace_stack
from .workspace import Workspace

_DEFAULT_MAX_FILES = 20


class GitStatusArguments(BaseModel):
    """Arguments for the ``git_status`` tool."""

    include_untracked: bool = Field(default=True)
    max_files: int = Field(default=_DEFAULT_MAX_FILES, gt=0, le=500)


def _current_workspace() -> Workspace:
    if not _workspace_stack:
        raise EditFileError(
            "git_status invoked without an active workspace; wrap the run in "
            "soteria_loop.app_tools.file_tools.bind_workspace(...)"
        )
    return _workspace_stack[-1]


def _summarise(status: GitStatus, arguments: GitStatusArguments) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "root": str(status.root),
        "branch": status.branch,
        "clean": status.clean,
        "modified_count": len(status.modified),
        "untracked_count": len(status.untracked),
    }
    if arguments.include_untracked:
        payload["untracked"] = list(status.untracked[: arguments.max_files])
    payload["modified"] = list(status.modified[: arguments.max_files])
    payload["truncated"] = len(status.modified) > arguments.max_files
    return payload


async def _git_status(arguments: GitStatusArguments) -> dict[str, Any]:
    workspace = _current_workspace()
    repo = GitRepository(workspace.root)
    status = repo.status()
    return _summarise(status, arguments)


def git_status_tool() -> PublicFunctionTool[GitStatusArguments]:
    """Return a :class:`FunctionTool` that surfaces git state."""

    return PublicFunctionTool(
        name="git_status",
        description=(
            "Return the git status of the active workspace — branch, "
            "modified files, and (optionally) untracked files. Read-only."
        ),
        arguments_model=GitStatusArguments,
        function=_git_status,
    )


__all__ = ["GitStatusArguments", "git_status_tool"]


_ = Path  # re-export for type checker
