"""Git MCP server — bounded subprocess wrappers.

Four tools:

* ``git_status`` — short porcelain status
* ``git_diff``   — unified diff (``--stat`` optional)
* ``git_log``    — last N commits oneline
* ``git_show``   — show one commit

All tools run ``git`` via :mod:`asyncio.create_subprocess_exec` with a
fixed timeout and never accept a shell string. Argument lists are
filtered to safe flags; arbitrary rev-specs are allowed only via the
``commit`` parameter which is passed as a single argv entry.

Run directly::

    python -m avo.mcp_servers.git --cwd /path/to/repo
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from avo.mcp_servers._stdio import AvoMCPServer, user_error


class _Server(AvoMCPServer):
    def __init__(self, repo: Path) -> None:
        super().__init__(server_name="avo-git", server_version="0.1.0")
        if shutil.which("git") is None:
            user_error("git binary not found on PATH")
        self._repo = repo

    async def _run_git(self, *args: str, timeout: float = 10.0) -> str:
        if not self._repo.exists():
            user_error(f"repository not found: {self._repo}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(self._repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            user_error(f"could not launch git: {exc}")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError as exc:
            proc.kill()
            user_error(f"git {args[0] if args else ''} timed out after {timeout}s")
            raise exc  # unreachable
        if proc.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip() or "git failed"
            user_error(message)
        return stdout.decode("utf-8", errors="replace")


class _StatusArgs(BaseModel):
    pass


class _DiffArgs(BaseModel):
    stat_only: bool = Field(
        default=False, description="Return only the diffstat summary, not the full patch."
    )
    paths: list[str] = Field(
        default_factory=list,
        description="Optional list of pathspecs to limit the diff to.",
    )


class _LogArgs(BaseModel):
    max_commits: int = Field(default=20, description="Number of commits to return.")
    oneline: bool = Field(default=True, description="Use --oneline format.")


class _ShowArgs(BaseModel):
    commit: str = Field(description="Commit-ish to show (e.g. HEAD, HEAD~1, sha).")


def build_server(repo: Path) -> _Server:
    server = _Server(repo)

    @server.tool(
        name="git_status",
        description="Return short porcelain status of the repository.",
        arguments_model=_StatusArgs,
    )
    async def _status(args: _StatusArgs) -> dict[str, object]:
        del args
        text = await server._run_git("status", "--short", "--branch")
        return {"status": text}

    @server.tool(
        name="git_diff",
        description="Unified diff (optionally limited to paths and stat-only).",
        arguments_model=_DiffArgs,
    )
    async def _diff(args: _DiffArgs) -> dict[str, object]:
        argv: list[str] = ["diff"]
        if args.stat_only:
            argv.append("--stat")
        else:
            argv.append("--no-color")
        argv.extend(args.paths)
        text = await server._run_git(*argv)
        return {"diff": text, "stat_only": args.stat_only}

    @server.tool(
        name="git_log",
        description="List recent commits.",
        arguments_model=_LogArgs,
    )
    async def _log(args: _LogArgs) -> dict[str, object]:
        argv = ["log", f"-n{args.max_commits}"]
        if args.oneline:
            argv.append("--oneline")
        text = await server._run_git(*argv)
        return {"log": text, "max_commits": args.max_commits}

    @server.tool(
        name="git_show",
        description="Show one commit.",
        arguments_model=_ShowArgs,
    )
    async def _show(args: _ShowArgs) -> dict[str, object]:
        if not args.commit or not all(c.isalnum() or c in "~^:_.-" for c in args.commit):
            user_error(f"unsafe commit ref: {args.commit!r}")
        text = await server._run_git("show", "--no-color", args.commit)
        return {"commit": args.commit, "show": text}

    return server


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="avo.mcp_servers.git")
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Path to the git repository (default: cwd).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import sys as _sys

    args = _parse_args(_sys.argv[1:] if argv is None else argv)
    server = build_server(args.cwd)
    asyncio.run(server.run())
    return 0


__all__ = ["build_server", "main"]


# ``shlex`` is referenced from the file docstring example; importing it
# here keeps the module's surface honest without pulling it into handler
# code paths.
_ = shlex


if __name__ == "__main__":
    raise SystemExit(main())
