"""Dependency-free command-line inspection for SQLite run databases."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from avo.chat import run_repl
from avo.doctor import main as doctor_main
from avo.exceptions import AvoError
from avo.providers.fake import FakeProvider
from avo.runtime import AgentRuntime
from avo.storage.sqlite import SQLiteEventStore
from avo.tracing import TraceInspector


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avo",
        description="Inspect, resume, and chat with Avo SQLite runs.",
    )
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=Path("avo.db"),
        help="SQLite database path (default: avo.db).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    runs = commands.add_parser("runs", help="Manage persisted runs.")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_commands.add_parser("list", help="List runs.")
    inspect_parser = run_commands.add_parser("inspect", help="Render one run trace.")
    inspect_parser.add_argument("run_id")
    resume_parser = run_commands.add_parser("resume", help="Resume a scripted fake-provider run.")
    resume_parser.add_argument("run_id")

    chat = commands.add_parser(
        "chat",
        help="Start an interactive chat REPL that drives one AgentRuntime turn per input.",
    )
    chat.add_argument(
        "--database",
        "-d",
        type=Path,
        default=Path("avo.db"),
        help="SQLite database path (default: avo.db).",
    )
    chat.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Workspace directory file tools are bound to (default: current working directory).",
    )
    chat.add_argument(
        "--session",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Resume an existing chat session by id (default: start a fresh thread; "
        "without --session the REPL offers to resume the most recent thread).",
    )
    chat.add_argument(
        "--new-session",
        action="store_true",
        help="Always start a fresh chat session, ignoring any prior threads.",
    )

    commands.add_parser(
        "doctor",
        help="Verify AVO_ provider configuration without making an HTTP call.",
    )

    return parser


async def _execute(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        # ``doctor_main`` has already-consumed argv; pass an empty list
        # so the inner argparse does not re-read sys.argv and complain
        # about the parent command's tail.
        return doctor_main([])

    if args.command == "chat":
        workspace_root = (args.workspace_root or Path.cwd()).resolve()
        db_path = args.database if args.database is not None else Path("avo.db")
        return await run_repl(
            database_path=db_path,
            workspace_root=workspace_root,
            session_id=args.session,
            force_new_session=args.new_session,
        )

    store = SQLiteEventStore(args.database)
    try:
        if args.runs_command == "list":
            runs = await store.list_runs()
            if not runs:
                print("No runs found.")
                return 0
            print(f"{'RUN ID':36}  {'STATE':20}  {'STOP REASON':24}  STEPS")
            for run in runs:
                reason = run.stop_reason.value if run.stop_reason is not None else "-"
                print(f"{run.run_id:36}  {run.state.value:20}  {reason:24}  {run.steps}")
            return 0

        if args.runs_command == "inspect":
            trace = await TraceInspector(store).inspect(args.run_id)
            print(trace.to_text())
            return 0

        if args.runs_command == "resume":
            checkpoint = await store.get_latest_checkpoint(args.run_id)
            if checkpoint is None:
                raise AvoError(f"Run {args.run_id!r} has no checkpoint and cannot be resumed.")
            if checkpoint.provider_metadata.get("provider_type") != "fake":
                raise AvoError(
                    "CLI resume can reconstruct only the built-in FakeProvider. "
                    "Resume real providers from application code with the configured adapter."
                )
            if (
                checkpoint.pending_response is not None
                and checkpoint.pending_response.tool_call is not None
            ):
                raise AvoError(
                    "CLI resume cannot reconstruct application tool callables. "
                    "Resume this run from application code with its ToolRegistry."
                )
            provider = FakeProvider.from_snapshot(checkpoint.provider_metadata)
            runtime = AgentRuntime(provider=provider, event_store=store)
            result = await runtime.resume(args.run_id)
            print(f"Run {result.run_id}: {result.status.value} ({result.stop_reason.value})")
            return 0
        raise AvoError(f"Unknown runs command: {args.runs_command!r}.")
    finally:
        await store.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Avo CLI and return a process exit status."""

    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_execute(args))
    except (AvoError, OSError) as exc:
        print(f"avo: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
