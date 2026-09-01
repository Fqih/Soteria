"""Workspace-bounded filesystem MCP server.

Exposes four tools over JSON-RPC stdio:

* ``read_file``        — return a file's contents (capped)
* ``write_file``       — write a file inside the workspace
* ``list_directory``   — list one directory
* ``search_files``     — recursive glob from the workspace root

Every path the server accepts is resolved against ``--root`` (default:
current directory). Paths that escape the root are refused with a
user-facing error so the MCP response is ``isError=True`` rather than a
transport-level failure.

Run directly::

    python -m avo.mcp_servers.filesystem --root /path/to/workspace
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path

from pydantic import BaseModel, Field

from avo.mcp_servers._stdio import AvoMCPServer, user_error

MAX_READ_BYTES = 1_048_576  # 1 MiB hard cap per read


class _Server(AvoMCPServer):
    def __init__(self, root: Path) -> None:
        super().__init__(server_name="avo-filesystem", server_version="0.1.0")
        self._root = root.resolve()

    def _resolve(self, raw: str) -> Path:
        candidate = (self._root / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            user_error(f"path {raw!r} escapes workspace root {self._root}")
            raise exc  # unreachable; user_error raises
        return candidate


class _ReadFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path of the file to read.")
    max_bytes: int = Field(
        default=MAX_READ_BYTES,
        description="Maximum bytes to return; defaults to 1 MiB.",
    )


class _WriteFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path of the file to write.")
    content: str = Field(description="UTF-8 text to write.")
    create_parents: bool = Field(
        default=False,
        description="Create parent directories if missing.",
    )


class _ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Workspace-relative directory to list.")
    max_entries: int = Field(default=500, description="Maximum entries to return.")


class _SearchArgs(BaseModel):
    pattern: str = Field(description="Glob pattern relative to workspace root (e.g. **/*.py).")
    max_entries: int = Field(default=500, description="Maximum entries to return.")


def build_server(root: Path) -> _Server:
    server = _Server(root)

    @server.tool(
        name="read_file",
        description="Read a UTF-8 text file from the workspace root.",
        arguments_model=_ReadFileArgs,
    )
    async def _read(args: _ReadFileArgs) -> dict[str, object]:
        target = server._resolve(args.path)
        if not target.exists() or not target.is_file():
            user_error(f"file not found: {args.path}")
        data = target.read_bytes()[: args.max_bytes]
        return {
            "path": str(target),
            "size": target.stat().st_size,
            "truncated": target.stat().st_size > len(data),
            "content": data.decode("utf-8", errors="replace"),
        }

    @server.tool(
        name="write_file",
        description="Write a UTF-8 text file inside the workspace.",
        arguments_model=_WriteFileArgs,
    )
    async def _write(args: _WriteFileArgs) -> dict[str, object]:
        target = server._resolve(args.path)
        if target.exists() and target.is_dir():
            user_error(f"{args.path!r} is a directory")
        if args.create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        elif not target.parent.exists():
            user_error(f"parent directory does not exist: {target.parent}")
        data = args.content.encode("utf-8")
        target.write_bytes(data)
        return {"path": str(target), "size": len(data)}

    @server.tool(
        name="list_directory",
        description="List entries in a workspace directory.",
        arguments_model=_ListDirArgs,
    )
    async def _list(args: _ListDirArgs) -> dict[str, object]:
        target = server._resolve(args.path or ".")
        if not target.exists() or not target.is_dir():
            user_error(f"directory not found: {args.path!r}")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        truncated = len(entries) > args.max_entries
        entries = entries[: args.max_entries]
        return {
            "path": str(target),
            "entries": [
                {
                    "name": entry.name,
                    "path": str(entry.relative_to(server._root)),
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
                for entry in entries
            ],
            "truncated": truncated,
        }

    @server.tool(
        name="search_files",
        description="Glob-search files relative to the workspace root.",
        arguments_model=_SearchArgs,
    )
    async def _search(args: _SearchArgs) -> dict[str, object]:
        matches: list[str] = []
        truncated = False
        for path in server._root.glob(args.pattern):
            rel = str(path.relative_to(server._root))
            if (matches and fnmatch.fnmatch(rel, args.pattern)) or matches is not None:
                matches.append(rel)
            if len(matches) >= args.max_entries:
                truncated = True
                break
        return {
            "pattern": args.pattern,
            "matches": matches[: args.max_entries],
            "truncated": truncated,
        }

    return server


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="avo.mcp_servers.filesystem")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root the server confines reads/writes to (default: cwd).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import asyncio
    import sys as _sys

    args = _parse_args(_sys.argv[1:] if argv is None else argv)
    server = build_server(args.root)
    asyncio.run(server.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
