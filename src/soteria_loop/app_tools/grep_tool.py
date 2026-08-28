"""``grep`` tool: find files whose contents match a regex.

The tool resolves ``path`` against the active :class:`Workspace`, walks
files (optionally restricted by ``include_glob``), and reports every
line that matches ``pattern``. ``context_lines`` adds N surrounding
lines per match so the model can read the matching region without a
second tool call.

The output is structured as a list of match records so the model can
distinguish file boundaries in a single response. Results are capped at
``max_results`` (default 100) so a broad regex cannot flood the
context window.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from soteria_loop import FunctionTool as PublicFunctionTool

from .edit_file import EditFileError  # workspace-binding contract
from .file_tools import _workspace_stack
from .workspace import Workspace

_DEFAULT_MAX_RESULTS = 100
_BINARY_SNIFF_BYTES = 4096


class GrepArguments(BaseModel):
    """Arguments for the ``grep`` tool."""

    pattern: str = Field(min_length=1, description="Python regex pattern")
    path: str = Field(default=".", description="Workspace-relative directory to search")
    include_glob: str | None = Field(
        default=None,
        description="Restrict the walk to files whose name matches this fnmatch pattern",
    )
    case_insensitive: bool = Field(default=False)
    max_results: int = Field(default=_DEFAULT_MAX_RESULTS, gt=0, le=10_000)
    context_lines: int = Field(default=0, ge=0, le=20)


def _current_grep_workspace() -> Workspace:
    if not _workspace_stack:
        raise EditFileError(
            "grep invoked without an active workspace; wrap the run in "
            "soteria_loop.app_tools.file_tools.bind_workspace(...)"
        )
    return _workspace_stack[-1]


def _iter_files(base: Path, include_glob: str | None) -> Iterable[Path]:
    """Yield text-like files under ``base`` matching ``include_glob``."""

    for candidate in base.rglob("*"):
        if not candidate.is_file():
            continue
        if include_glob is not None and not fnmatch(candidate.name, include_glob):
            continue
        try:
            with candidate.open("rb") as handle:
                head = handle.read(_BINARY_SNIFF_BYTES)
        except OSError:
            continue
        if b"\x00" in head:
            continue  # binary file
        yield candidate


def _render_context(
    lines: list[str],
    line_number: int,
    context_lines: int,
) -> dict[str, list[str]]:
    """Return ``before``/``after`` slices around ``line_number`` (1-indexed)."""

    start = max(0, line_number - 1 - context_lines)
    end = min(len(lines), line_number - 1 + context_lines + 1)
    return {
        "before": lines[start : line_number - 1],
        "after": lines[line_number:end],
    }


async def _grep(arguments: GrepArguments) -> dict[str, Any]:
    workspace = _current_grep_workspace()
    base = workspace.validate_path(arguments.path, must_exist=True)
    if not base.is_dir():
        raise EditFileError(f"grep path is not a directory: {arguments.path}")

    flags = re.MULTILINE
    if arguments.case_insensitive:
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(arguments.pattern, flags)
    except re.error as exc:
        raise EditFileError(f"invalid regex pattern: {exc}") from exc

    matches: list[dict[str, Any]] = []
    truncated = False
    files_scanned = 0

    for file_path in _iter_files(base, arguments.include_glob):
        files_scanned += 1
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = content.splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not compiled.search(line):
                continue
            record: dict[str, Any] = {
                "path": file_path.resolve().relative_to(workspace.root).as_posix(),
                "line_number": line_number,
                "line": line,
            }
            if arguments.context_lines > 0:
                record["context"] = _render_context(lines, line_number, arguments.context_lines)
            matches.append(record)
            if len(matches) >= arguments.max_results:
                truncated = True
                break
        if truncated:
            break

    return {
        "pattern": arguments.pattern,
        "base": str(base),
        "files_scanned": files_scanned,
        "count": len(matches),
        "truncated": truncated,
        "matches": matches,
    }


def grep_tool() -> PublicFunctionTool[GrepArguments]:
    """Return a :class:`FunctionTool` that searches file contents."""

    return PublicFunctionTool(
        name="grep",
        description=(
            "Search workspace files for a regex pattern. Restrict with "
            "``include_glob`` and add ``context_lines`` for surrounding "
            "lines. Results are bounded by ``max_results``; binary files "
            "are skipped automatically."
        ),
        arguments_model=GrepArguments,
        function=_grep,
    )


__all__ = ["GrepArguments", "grep_tool"]
