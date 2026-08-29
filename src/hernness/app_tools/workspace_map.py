"""``workspace_map`` tool — expose the workspace index to the model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from hernness import FunctionTool as PublicFunctionTool

from ..workspace.indexer import WorkspaceIndexer
from .edit_file import EditFileError
from .file_tools import _workspace_stack
from .workspace import Workspace

_DEFAULT_MAX_ENTRIES = 200
_DEFAULT_RECENT_LIMIT = 20


class WorkspaceMapArguments(BaseModel):
    """Arguments for the ``workspace_map`` tool."""

    include_map: bool = Field(default=True)
    include_recent: bool = Field(default=True)
    max_entries: int = Field(default=_DEFAULT_MAX_ENTRIES, gt=0, le=5_000)
    recent_limit: int = Field(default=_DEFAULT_RECENT_LIMIT, gt=0, le=200)


def _current_workspace() -> Workspace:
    if not _workspace_stack:
        raise EditFileError(
            "workspace_map invoked without an active workspace; wrap the run in "
            "hernness.app_tools.file_tools.bind_workspace(...)"
        )
    return _workspace_stack[-1]


async def _workspace_map(arguments: WorkspaceMapArguments) -> dict[str, Any]:
    workspace = _current_workspace()
    index = WorkspaceIndexer(workspace.root).build(max_entries=arguments.max_entries)
    payload: dict[str, Any] = {
        "root": str(workspace.root),
        "entry_count": len(index.entries),
        "truncated": index.truncated,
    }
    if arguments.include_map:
        payload["map"] = index.map_text(max_entries=arguments.max_entries)
    if arguments.include_recent:
        payload["recent"] = [
            entry.to_dict() for entry in index.recent_files(limit=arguments.recent_limit)
        ]
    return payload


def workspace_map_tool() -> PublicFunctionTool[WorkspaceMapArguments]:
    """Return a :class:`FunctionTool` that maps the workspace."""

    return PublicFunctionTool(
        name="workspace_map",
        description=(
            "Return a snapshot of the active workspace: a compact file map "
            "and the most-recently-modified files. Read-only."
        ),
        arguments_model=WorkspaceMapArguments,
        function=_workspace_map,
    )


__all__ = ["WorkspaceMapArguments", "workspace_map_tool"]
