"""Interactive chat REPL wiring AgentRuntime + app_tools + SQLite."""

from __future__ import annotations

import getpass
import os
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from soteria_loop.app_tools.file_tools import bind_workspace, read_file_tool, write_file_tool
from soteria_loop.app_tools.workspace import Workspace
from soteria_loop.config import ConfigError, build_provider_from_env
from soteria_loop.exceptions import SoteriaError
from soteria_loop.runtime import AgentRuntime
from soteria_loop.storage.sqlite import SQLiteEventStore
from soteria_loop.tracing import TraceInspector

REPO_LOGO_PATH = Path(__file__).resolve().parents[3] / "public" / "logo.webp"


# Provider catalog used by the interactive first-run setup. Each entry:
#   key      -> canonical provider name (matches SOTERIA_PROVIDER)
#   label    -> human-friendly label printed in the menu
#   default_model -> suggested model if the user accepts the default
#   needs_api_key  -> True for hosted providers; False for local Ollama
_PROVIDER_CATALOG: dict[str, dict[str, str | bool]] = {
    "ollama": {
        "label": "Ollama",
        "default_model": "llama3.1",
        "needs_api_key": False,
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-5.6",
        "needs_api_key": True,
    },
    "anthropic": {
        "label": "Anthropic",
        "default_model": "claude-sonnet-4-6",
        "needs_api_key": True,
    },
    "minimax": {
        "label": "MiniMax",
        "default_model": "MiniMax-M3",
        "needs_api_key": True,
    },
}


_FIRST_RUN_MESSAGE = (
    "Soteria is not configured yet.\n"
    "\n"
    "No provider has been configured.\n"
    "\n"
    "Configure one of:\n"
    "\n"
    "  SOTERIA_PROVIDER=ollama\n"
    "  SOTERIA_PROVIDER=minimax\n"
    "  SOTERIA_PROVIDER=anthropic\n"
    "  SOTERIA_PROVIDER=openai\n"
    "\n"
    "with the matching provider-specific keys and model. See .env.example "
    "for the full list of recognised variables.\n"
    "\n"
    "Then retry:\n"
    "\n"
    "  soteria-loop chat\n"
)


@dataclass
class ChatContext:
    """Everything the REPL keeps alive across turns."""

    runtime: AgentRuntime
    store: SQLiteEventStore
    workspace: Workspace
    provider_name: str
    model_name: str


def _read_environ() -> dict[str, str]:
    """Snapshot ``os.environ`` so the chat does not see mid-session mutations."""

    return dict(os.environ)


def _resolve_provider_label(environ: dict[str, str]) -> tuple[str, str]:
    """Return ``(provider_name, model_name)`` without exposing secrets."""

    provider = environ.get("SOTERIA_PROVIDER", "").strip() or "(unset)"
    model = (
        environ.get("SOTERIA_MODEL")
        or environ.get("MODEL_MINIMAX")
        or environ.get("OPENAI_MODEL")
        or "(unset)"
    )
    return provider, model


def _print_header(out: TextIO, ctx: ChatContext, workspace_root: Path) -> None:
    out.write("Soteria\n")
    out.write(f"Provider: {ctx.provider_name}\n")
    out.write(f"Model: {ctx.model_name}\n")
    out.write(f"Workspace: {workspace_root}\n")
    out.write("Logo: " + str(REPO_LOGO_PATH) + "\n")
    out.write("Slash commands: /provider, /inspect RUN_ID, /resume RUN_ID, /quit\n")
    out.write("Enter a task to run one AgentRuntime turn. Ctrl+D or /quit to exit.\n")
    out.flush()


def _print_provider_summary(out: TextIO, ctx: ChatContext, environ: dict[str, str]) -> None:
    """Display provider/model without leaking API keys or tokens."""

    out.write(f"Provider: {ctx.provider_name}\n")
    out.write(f"Model: {ctx.model_name}\n")
    base_url = environ.get(f"SOTERIA_{ctx.provider_name.upper()}_BASE_URL", "") or "(default)"
    out.write(f"Base URL: {base_url}\n")
    has_key = bool(environ.get(f"SOTERIA_{ctx.provider_name.upper()}_API_KEY", "").strip())
    out.write(f"API key configured: {has_key}\n")
    out.flush()


def build_chat_context(
    *,
    database_path: Path,
    workspace_root: Path,
    environ: dict[str, str],
) -> ChatContext:
    """Construct the runtime + store + workspace bound together."""

    provider_name, model_name = _resolve_provider_label(environ)
    provider = build_provider_from_env(environ)
    workspace = Workspace(workspace_root, create=False)
    store = SQLiteEventStore(database_path)
    runtime = AgentRuntime(
        provider=provider,
        event_store=store,
        tools=[read_file_tool(), write_file_tool()],
    )
    return ChatContext(
        runtime=runtime,
        store=store,
        workspace=workspace,
        provider_name=provider_name,
        model_name=model_name,
    )


def render_first_run_message(out: TextIO) -> None:
    """Print the canonical first-run configuration hint."""

    out.write(_FIRST_RUN_MESSAGE)
    out.flush()


# ---------------------------------------------------------------------------
# Interactive first-run setup
# ---------------------------------------------------------------------------


class _SetupAborted(Exception):
    """Raised when the operator aborts the interactive first-run setup."""


def _prompt_line(in_stream: TextIO, out_stream: TextIO, prompt: str) -> str:
    """Read one line from ``in_stream`` after printing ``prompt`` to ``out_stream``.

    Strips the trailing newline. Returns an empty string on EOF.
    Raises :class:`_SetupAborted` on EOF.
    """

    out_stream.write(prompt)
    out_stream.flush()
    line = in_stream.readline()
    if not line:
        raise _SetupAborted
    return line.rstrip("\n").rstrip("\r")


def _prompt_choice(in_stream: TextIO, out_stream: TextIO, prompt: str, choices: list[str]) -> str:
    """Prompt until the operator picks one of ``choices`` (case-insensitive).

    Re-prompts on invalid input. Returns the canonical lowercase choice.
    """

    lowered = {c.lower(): c for c in choices}
    while True:
        raw = _prompt_line(in_stream, out_stream, prompt)
        if raw.strip().lower() in lowered:
            return lowered[raw.strip().lower()]
        out_stream.write(f"please choose one of: {', '.join(choices)}\n")
        out_stream.flush()


def _prompt_optional(
    in_stream: TextIO,
    out_stream: TextIO,
    prompt: str,
    *,
    default: str | None = None,
) -> str:
    """Prompt with optional default. Empty input -> default (or empty)."""

    suffix = f" [{default}]" if default else ""
    raw = _prompt_line(in_stream, out_stream, f"{prompt}{suffix}: ").strip()
    if not raw:
        return default or ""
    return raw


def _prompt_required(
    in_stream: TextIO,
    out_stream: TextIO,
    prompt: str,
    *,
    secret: bool = False,
    secret_reader: Callable[[str], str] | None = None,
) -> str:
    """Prompt until the operator supplies a non-empty value.

    When ``secret=True`` the value is read via ``secret_reader`` (defaults
    to :func:`getpass.getpass`) so it is never echoed to the terminal.
    """

    reader = secret_reader or (lambda p: getpass.getpass(p))
    while True:
        value = reader(prompt).strip()
        if value:
            return value
        out_stream.write("value cannot be empty\n")
        out_stream.flush()


def interactive_first_run_setup(
    stdin: TextIO,
    stdout: TextIO,
    *,
    secret_reader: Callable[[str], str] | None = None,
) -> dict[str, str] | None:
    """Prompt the operator through provider, API key, and model selection.

    Returns a dict of env-style variables (``SOTERIA_PROVIDER``,
    ``SOTERIA_<PROVIDER>_API_KEY``, ``SOTERIA_MODEL``, ...) ready to be
    merged into the chat REPL environment. Returns ``None`` if the
    operator aborts (Ctrl+C / Ctrl+D).

    The function never persists secrets to disk. The returned dict lives
    only inside the running process so the API key disappears with the
    process unless the operator also exports it in their shell.
    """

    try:
        stdout.write("\n")
        stdout.write("Soteria First-Time Setup\n")
        stdout.write("\n")
        stdout.write("No AI provider has been configured yet.\n")
        stdout.write("\n")
        stdout.write("Select your provider:\n")
        stdout.write("\n")
        for index, key in enumerate(_PROVIDER_CATALOG, start=1):
            label = _PROVIDER_CATALOG[key]["label"]
            stdout.write(f"  {index}. {label}\n")
        stdout.write("\n")

        keys = list(_PROVIDER_CATALOG.keys())
        display_choices = [str(i) for i in range(1, len(keys) + 1)]
        chosen_display = _prompt_choice(
            stdin,
            stdout,
            f"Select provider [1-{len(keys)}]: ",
            display_choices,
        )
        provider_key = keys[int(chosen_display) - 1]
        spec = _PROVIDER_CATALOG[provider_key]
        provider_label = str(spec["label"])

        env: dict[str, str] = {"SOTERIA_PROVIDER": provider_key}

        if spec["needs_api_key"]:
            api_key = _prompt_required(
                stdin,
                stdout,
                "API Key: ",
                secret=True,
                secret_reader=secret_reader,
            )
            env[f"SOTERIA_{provider_key.upper()}_API_KEY"] = api_key

        # Provider-specific optional tweaks.
        if provider_key == "openai":
            base_url = _prompt_optional(stdin, stdout, "OpenAI base URL (optional)", default="")
            if base_url:
                env["SOTERIA_OPENAI_BASE_URL"] = base_url
        elif provider_key == "minimax":
            style = _prompt_optional(
                stdin,
                stdout,
                "MiniMax API style (anthropic or openai)",
                default="anthropic",
            ).lower()
            if style not in ("anthropic", "openai"):
                stdout.write("invalid style; defaulting to anthropic\n")
                style = "anthropic"
            env["SOTERIA_MINIMAX_API_STYLE"] = style

        default_model = str(spec["default_model"])
        model = _prompt_optional(stdin, stdout, "Model", default=default_model).strip()
        env["SOTERIA_MODEL"] = model or default_model

        stdout.write(f"\nProvider configured: {provider_label} ({env['SOTERIA_MODEL']})\n")
        stdout.write("\nStarting Soteria...\n\n")
        stdout.flush()
        return env
    except _SetupAborted:
        stdout.write("\nsetup aborted; no changes applied.\n")
        stdout.flush()
        return None
    except KeyboardInterrupt:
        stdout.write("\nsetup aborted (Ctrl+C); no changes applied.\n")
        stdout.flush()
        return None


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
        except SoteriaError as exc:
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
        except SoteriaError as exc:
            err.write(f"resume failed: {exc}\n")
            return False
        out.write(
            f"Resumed run {result.run_id}: status={result.status.value} "
            f"stop_reason={result.stop_reason.value} steps={result.steps}\n"
        )
        if result.output:
            out.write(f"output: {result.output}\n")
        return False

    err.write(f"unknown command: {cmd}\n")
    return False


async def _run_turn(ctx: ChatContext, task: str, out: TextIO, err: TextIO) -> None:
    """Execute one user turn against ``ctx.runtime``."""

    with bind_workspace(ctx.workspace):
        try:
            result = await ctx.runtime.run(task)
        except SoteriaError as exc:
            err.write(f"runtime error: {exc}\n")
            return
        except Exception as exc:
            err.write(f"unexpected error: {type(exc).__name__}: {exc}\n")
            return

    out.write(
        f"Soteria [{result.status.value}/{result.stop_reason.value}] "
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
        # First-run UX: ask the operator to configure a provider
        # interactively. We never echo the ConfigError text so a
        # misconfigured SOTERIA_PROVIDER value cannot leak.
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
        except (SoteriaError, OSError) as exc:
            err_stream.write(f"soteria-loop chat: {exc}\n")
            return 2
        except ConfigError as exc:
            err_stream.write(f"configuration still invalid after setup: {exc}\n")
            return 2
    except (SoteriaError, OSError) as exc:
        err_stream.write(f"soteria-loop chat: {exc}\n")
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
    "build_chat_context",
    "interactive_first_run_setup",
    "render_first_run_message",
    "run_repl",
]
