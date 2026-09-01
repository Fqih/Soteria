"""``avo sandbox`` — standalone docker sandbox executor CLI.

Exposes the same sandbox primitive that :mod:`avo.app_tools.shell_tool`
uses internally, but as an operator-facing CLI so the sandbox can be
used outside the agent loop:

- ``avo sandbox run --image IMG --workspace DIR -- COMMAND [ARGS...]``

The CLI wraps :class:`avo.app_tools.sandbox.SandboxExecutor` and
prints a JSON report on completion. The docker client is injectable
through ``--client`` for tests; production uses ``docker.from_env()``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from avo.exceptions import AvoError


class SandboxCliError(AvoError):
    """User-facing failure in ``avo sandbox``."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avo sandbox",
        description="Run a shell command inside an ephemeral docker sandbox.",
    )
    subparsers = parser.add_subparsers(dest="sandbox_command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a single shell command inside an ephemeral sandbox container.",
    )
    run_parser.add_argument(
        "--image",
        default="python:3.12-slim",
        help="Docker image used for the sandbox container (default: python:3.12-slim).",
    )
    run_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory for the sandbox (default: current working directory).",
    )
    run_parser.add_argument(
        "--network",
        default="none",
        help="Docker network mode (default: none — fully offline).",
    )
    run_parser.add_argument(
        "--memory",
        default="256m",
        help="Memory limit passed to docker (default: 256m).",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Hard wall-clock cap in seconds (default: 30).",
    )
    run_parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to inject into the container (repeatable).",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report (default: human-readable).",
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Shell command to execute inside the sandbox.",
    )
    return parser


async def _invoke(args: argparse.Namespace) -> int:
    from avo.app_tools.sandbox import SandboxExecutor

    command_parts = list(args.command)
    # Strip a leading "--" separator if the caller passed one.
    if command_parts and command_parts[0] == "--":
        command_parts = command_parts[1:]
    if not command_parts:
        raise SandboxCliError("no command provided; pass a shell command after the flags.")
    command = " ".join(command_parts)

    env: dict[str, str] = {}
    for entry in args.env:
        if "=" not in entry:
            raise SandboxCliError(f"--env entries must look like KEY=VALUE; got {entry!r}")
        key, value = entry.split("=", 1)
        env[key] = value

    executor = SandboxExecutor(
        image=args.image,
        mem_limit=args.memory,
        network_mode=args.network,
        timeout_seconds=args.timeout,
    )
    workspace = args.workspace.resolve()
    result = await executor.run(command, workspace_dir=workspace, env=env)
    if args.json:
        sys.stdout.write(
            json.dumps(
                {
                    "command": command,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration_ms": result.duration_ms,
                    "image": result.image,
                    "network_mode": result.network_mode,
                    "mem_limit": result.mem_limit,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stdout.write(result.stderr)
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.sandbox_command == "run":
        return asyncio.run(_invoke(args))
    raise SandboxCliError(f"Unknown subcommand: {args.sandbox_command!r}")


__all__ = ["SandboxCliError", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
