"""``glob`` tool: enumerate workspace files matching a pattern.

The tool resolves ``path`` against the active :class:`Workspace`,
rejects any traversal, then walks with :meth:`pathlib.Path.glob` (or
:meth:`rglob` when ``recursive=True``). Results are returned as paths
relative to the workspace root so the model never sees absolute
locations outside the workspace.

Output is sorted and capped at ``max_results`` (default 200) so a
generous pattern cannot flood the context window.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from avo import FunctionTool as PublicFunctionTool

from .edit_file import EditFileError  # reuse the workspace-binding contract
from .file_tools import _workspace_stack
from .workspace import Workspace

_DEFAULT_MAX_RESULTS = 200


class GlobArguments(BaseModel):
    """Arguments for the ``glob`` tool."""

    pattern: str = Field(min_length=1, description="Glob pattern, e.g. ``**/*.py``")
    path: str = Field(
        default=".",
        description="Workspace-relative or absolute directory to search in",
    )
    recursive: bool = Field(default=True)
    max_results: int = Field(default=_DEFAULT_MAX_RESULTS, gt=0, le=10_000)


def _current_glob_workspace() -> Workspace:
    if not _workspace_stack:
        raise EditFileError(
            "glob invoked without an active workspace; wrap the run in "
            "avo.app_tools.file_tools.bind_workspace(...)"
        )
    return _workspace_stack[-1]


async def _glob(arguments: GlobArguments) -> dict[str, Any]:
    workspace = _current_glob_workspace()
    base = workspace.validate_path(arguments.path, must_exist=True)
    if not base.is_dir():
        raise EditFileError(f"glob path is not a directory: {arguments.path}")

    walker = base.rglob if arguments.recursive else base.glob
    matches = sorted(walker(arguments.pattern))

    relative: list[str] = []
    truncated = False
    for candidate in matches:
        if not candidate.is_file() and not candidate.is_dir():
            continue
        try:
            relative_path = candidate.resolve().relative_to(workspace.root)
        except ValueError as exc:  # pragma: no cover - guarded by validate_path
            raise EditFileError(f"escape detected: {candidate}") from exc
        if len(relative) >= arguments.max_results:
            truncated = True
            break
        relative.append(relative_path.as_posix())

    return {
        "pattern": arguments.pattern,
        "base": str(base),
        "count": len(relative),
        "truncated": truncated,
        "matches": relative,
    }


def glob_tool() -> PublicFunctionTool[GlobArguments]:
    """Return a :class:`FunctionTool` that enumerates workspace files."""

    return PublicFunctionTool(
        name="glob",
        description=(
            "List files inside the workspace matching a glob pattern (e.g. "
            "``**/*.py``). Paths are returned relative to the workspace root; "
            "any traversal that escapes the root is rejected before the walk."
        ),
        arguments_model=GlobArguments,
        function=_glob,
    )


__all__ = ["GlobArguments", "glob_tool"]
