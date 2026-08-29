"""Interactive chat REPL wiring AgentRuntime + app_tools + SQLite.

The setup wizard lives in :mod:`hernness.chat_setup` and the
shell-rc persistence helpers live in :mod:`hernness.chat_shell_rc`.
Both modules are re-exported here for backward compatibility with
existing imports (``from hernness.chat import interactive_first_run_setup``)
so the public surface stays stable.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from hernness import runtime as _runtime  # noqa: F401  (typing hook)
from hernness.app_tools.file_tools import bind_workspace, read_file_tool, write_file_tool
from hernness.app_tools.workspace import Workspace
from hernness.chat_setup import interactive_first_run_setup  # re-export
from hernness.chat_shell_rc import (  # re-export
    _detect_shell_rc_path,
    _offer_persist_to_shell_rc,
    _quote_for_shell,
    persist_env_to_shell_rc,
)
from hernness.config import ConfigError, build_provider_from_env
from hernness.exceptions import HernnessError
from hernness.runtime import AgentRuntime
from hernness.skills import SkillRegistry
from hernness.storage.sqlite import SQLiteEventStore
from hernness.tracing import TraceInspector

REPO_LOGO_PATH = Path(__file__).resolve().parents[3] / "public" / "logo.webp"

_FIRST_RUN_MESSAGE = (
    "Hernness is not configured yet.\n"
    "\n"
    "No provider has been configured.\n"
    "\n"
    "Configure one of:\n"
    "\n"
    "  HERNNESS_PROVIDER=ollama\n"
    "  HERNNESS_PROVIDER=minimax\n"
    "  HERNNESS_PROVIDER=anthropic\n"
    "  HERNNESS_PROVIDER=openai\n"
    "\n"
    "with the matching provider-specific keys and model. See .env.example "
    "for the full list of recognised variables.\n"
    "\n"
    "Then retry:\n"
    "\n"
    "  hernness chat\n"
)


@dataclass
class ChatContext:
    """Everything the REPL keeps alive across turns."""

    runtime: AgentRuntime
    store: SQLiteEventStore
    workspace: Workspace
    provider_name: str
    model_name: str
    skills: SkillRegistry


def _read_environ() -> dict[str, str]:
    """Snapshot ``os.environ`` so the chat does not see mid-session mutations."""

    return dict(os.environ)


def _resolve_provider_label(environ: dict[str, str]) -> tuple[str, str]:
    """Return ``(provider_name, model_name)`` without exposing secrets."""

    provider = environ.get("HERNNESS_PROVIDER", "").strip() or "(unset)"
    model = (
        environ.get("HERNNESS_MODEL")
        or environ.get("MODEL_MINIMAX")
        or environ.get("OPENAI_MODEL")
        or "(unset)"
    )
    return provider, model


def _print_header(out: TextIO, ctx: ChatContext, workspace_root: Path) -> None:
    out.write("Hernness\n")
    out.write(f"Provider: {ctx.provider_name}\n")
    out.write(f"Model: {ctx.model_name}\n")
    out.write(f"Workspace: {workspace_root}\n")
    out.write("Logo: " + str(REPO_LOGO_PATH) + "\n")
    out.write(
        "Slash commands: /provider, /inspect RUN_ID, /resume RUN_ID, /skills, /skill NAME, /quit\n"
    )
    out.write("Enter a task to run one AgentRuntime turn. Ctrl+D or /quit to exit.\n")
    out.flush()


def _print_provider_summary(out: TextIO, ctx: ChatContext, environ: dict[str, str]) -> None:
    """Display provider/model without leaking API keys or tokens."""

    out.write(f"Provider: {ctx.provider_name}\n")
    out.write(f"Model: {ctx.model_name}\n")
    base_url = environ.get(f"HERNNESS_{ctx.provider_name.upper()}_BASE_URL", "") or "(default)"
    out.write(f"Base URL: {base_url}\n")
    has_key = bool(environ.get(f"HERNNESS_{ctx.provider_name.upper()}_API_KEY", "").strip())
    out.write(f"API key configured: {has_key}\n")
    out.flush()


def build_chat_context(
    *,
    database_path: Path,
    workspace_root: Path,
    environ: dict[str, str],
) -> ChatContext:
    """Construct the runtime + store + workspace bound together.

    Raises ``HernnessError`` (or a subclass) if ``database_path`` or
    ``workspace_root`` are unusable. ``ConfigError`` propagates unchanged
    from :func:`build_provider_from_env`.
    """

    if database_path is None:
        raise HernnessError("database_path must be a Path, not None; the CLI is misconfigured.")
    db_path = Path(database_path)

    provider_name, model_name = _resolve_provider_label(environ)
    provider = build_provider_from_env(environ)
    workspace = Workspace(workspace_root, create=False)
    store = SQLiteEventStore(db_path)
    runtime = AgentRuntime(
        provider=provider,
        event_store=store,
        tools=[read_file_tool(), write_file_tool()],
    )
    skills_root = workspace_root / ".soteria" / "skills"
    # Ensure the skills directory exists for first-run use, but the
    # registry itself walks a path — body lookup happens lazily.
    skills_root.mkdir(parents=True, exist_ok=True)
    skills = SkillRegistry(skills_root)
    return ChatContext(
        runtime=runtime,
        store=store,
        workspace=workspace,
        provider_name=provider_name,
        model_name=model_name,
        skills=skills,
    )


def render_first_run_message(out: TextIO) -> None:
    """Print the canonical first-run configuration hint."""

    out.write(_FIRST_RUN_MESSAGE)
    out.flush()


async def _run_slash(
    ctx: ChatContext,
    args: list[str],
    out: TextIO,
    err: TextIO,
    environ: dict[str, str],
) -> bool:
    """Dispatch a slash command. Returns True if the REPL should exit."""

    if not args:
        err.write("expected a slash command (try /provider)\n")
        return False

    cmd = args[0]

    if cmd in ("/quit", "/exit"):
        return True

    if cmd == "/provider":
        _print_provider_summary(out, ctx, environ)
        return False

    if cmd == "/inspect":
        if len(args) != 2:
            err.write("usage: /inspect RUN_ID\n")
            return False
        try:
            trace = await TraceInspector(ctx.store).inspect(args[1])
        except HernnessError as exc:
            err.write(f"inspect failed: {exc}\n")
            return False
        out.write(trace.to_text())
        out.write("\n")
        return False

    if cmd == "/resume":
        if len(args) != 2:
            err.write("usage: /resume RUN_ID\n")
            return False
        try:
            result = await ctx.runtime.resume(args[1])
        except HernnessError as exc:
            err.write(f"resume failed: {exc}\n")
            return False
        out.write(
            f"Resumed run {result.run_id}: status={result.status.value} "
            f"stop_reason={result.stop_reason.value} steps={result.steps}\n"
        )
        if result.output:
            out.write(f"output: {result.output}\n")
        return False

    if cmd == "/skills":
        names = ctx.skills.names()
        if not names:
            out.write(f"No skills found under {ctx.skills.root}\n")
            return False
        for name in names:
            out.write(f"  {name}\n")
        return False

    if cmd == "/skill":
        if len(args) != 2:
            err.write("usage: /skill NAME\n")
            return False
        try:
            body = ctx.skills.load(args[1])
        except HernnessError as exc:
            err.write(f"skill load failed: {exc}\n")
            return False
        # Skill bodies are injected as a user message so the runtime
        # treats them like any other turn input — no separate channel.
        await _run_turn(ctx, body, out, err)
        return False

    err.write(f"unknown command: {cmd}\n")
    return False


async def _run_turn(ctx: ChatContext, task: str, out: TextIO, err: TextIO) -> None:
    """Execute one user turn against ``ctx.runtime``."""

    with bind_workspace(ctx.workspace):
        try:
            result = await ctx.runtime.run(task)
        except HernnessError as exc:
            err.write(f"runtime error: {exc}\n")
            return
        except Exception as exc:
            err.write(f"unexpected error: {type(exc).__name__}: {exc}\n")
            return

    out.write(
        f"Hernness [{result.status.value}/{result.stop_reason.value}] "
        f"steps={result.steps} run_id={result.run_id}\n"
    )
    if result.error:
        out.write(f"error: {result.error}\n")
    if result.output:
        out.write(f"> {result.output}\n")
    out.flush()


async def run_repl(
    *,
    database_path: Path,
    workspace_root: Path,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environ: dict[str, str] | None = None,
    prompt: str = "You > ",
    secret_reader: Callable[[str], str] | None = None,
) -> int:
    """Run the interactive chat REPL until EOF, /quit, or fatal init error."""

    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout
    err_stream = stderr or sys.stderr
    env = environ if environ is not None else _read_environ()

    try:
        ctx = build_chat_context(
            database_path=database_path,
            workspace_root=workspace_root,
            environ=env,
        )
    except ConfigError:
        new_env = interactive_first_run_setup(in_stream, out_stream, secret_reader=secret_reader)
        if new_env is None:
            return 2
        env = {**env, **new_env}
        try:
            ctx = build_chat_context(
                database_path=database_path,
                workspace_root=workspace_root,
                environ=env,
            )
        except (HernnessError, OSError) as exc:
            err_stream.write(f"hernness chat: {exc}\n")
            return 2
        except ConfigError as exc:
            err_stream.write(f"configuration still invalid after setup: {exc}\n")
            return 2
    except (HernnessError, OSError) as exc:
        err_stream.write(f"hernness chat: {exc}\n")
        return 2

    _print_header(out_stream, ctx, workspace_root)
    out_stream.write("\n")
    out_stream.flush()

    try:
        while True:
            try:
                out_stream.write(prompt)
                out_stream.flush()
                line = in_stream.readline()
            except KeyboardInterrupt:
                out_stream.write("\n(interrupted - type /quit or Ctrl+D to exit)\n")
                out_stream.flush()
                continue

            if not line:
                out_stream.write("\n")
                return 0
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("/"):
                try:
                    args = shlex.split(stripped)
                except ValueError as exc:
                    err_stream.write(f"parse error: {exc}\n")
                    continue
                should_exit = await _run_slash(ctx, args, out_stream, err_stream, env)
                if should_exit:
                    return 0
                continue

            await _run_turn(ctx, stripped, out_stream, err_stream)
    finally:
        await ctx.store.close()


__all__ = [
    "REPO_LOGO_PATH",
    "ChatContext",
    "_detect_shell_rc_path",
    "_offer_persist_to_shell_rc",
    "_print_header",
    "_print_provider_summary",
    "_quote_for_shell",
    "_read_environ",
    "_resolve_provider_label",
    "_run_slash",
    "_run_turn",
    "build_chat_context",
    "interactive_first_run_setup",
    "persist_env_to_shell_rc",
    "render_first_run_message",
    "run_repl",
]
