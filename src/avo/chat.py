"""Interactive chat REPL wiring AgentRuntime + app_tools + SQLite.

The setup wizard lives in :mod:`avo.chat_setup` and the
shell-rc persistence helpers live in :mod:`avo.chat_shell_rc`.
Both modules are re-exported here for backward compatibility with
existing imports (``from avo.chat import interactive_first_run_setup``)
so the public surface stays stable.

Conversation threading lives in :mod:`avo.chat_session`. The
REPL persists every user input + assistant reply through
:class:`avo.storage.conversations.ConversationStore`, sharing the
SQLite file with the event store so a single file holds both the run
history and the chat thread.
"""

from __future__ import annotations

import os
import shlex
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from avo import __version__ as AVO_VERSION
from avo import runtime as _runtime  # noqa: F401  (typing hook)
from avo.app_tools.file_tools import bind_workspace, read_file_tool, write_file_tool
from avo.app_tools.workspace import Workspace
from avo.chat_session import (
    SessionInfo,
    SessionLifecycle,
    render_session_picker,
    render_session_row,
    resolve_session_id,
)
from avo.chat_setup import interactive_first_run_setup  # re-export
from avo.chat_shell_rc import (  # re-export
    _detect_shell_rc_path,
    _offer_persist_to_shell_rc,
    _quote_for_shell,
    persist_env_to_shell_rc,
)
from avo.config import ConfigError, build_provider_from_env
from avo.exceptions import AvoError
from avo.runtime import AgentRuntime
from avo.skills import SkillRegistry
from avo.storage.sqlite import SQLiteEventStore
from avo.tracing import TraceInspector

REPO_LOGO_PATH = Path(__file__).resolve().parents[3] / "public" / "logo.webp"

_FIRST_RUN_MESSAGE = (
    "Avo is not configured yet.\n"
    "\n"
    "No provider has been configured.\n"
    "\n"
    "Configure one of:\n"
    "\n"
    "  AVO_PROVIDER=ollama\n"
    "  AVO_PROVIDER=minimax\n"
    "  AVO_PROVIDER=anthropic\n"
    "  AVO_PROVIDER=openai\n"
    "\n"
    "with the matching provider-specific keys and model. See .env.example "
    "for the full list of recognised variables.\n"
    "\n"
    "Then retry:\n"
    "\n"
    "  avo chat\n"
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
    session: SessionLifecycle
    session_id: str
    pending_preamble: str | None = None


def _read_environ() -> dict[str, str]:
    """Snapshot ``os.environ`` so the chat does not see mid-session mutations."""

    return dict(os.environ)


def _resolve_provider_label(environ: dict[str, str]) -> tuple[str, str]:
    """Return ``(provider_name, model_name)`` without exposing secrets."""

    provider = environ.get("AVO_PROVIDER", "").strip() or "(unset)"
    model = (
        environ.get("AVO_MODEL")
        or environ.get("MODEL_MINIMAX")
        or environ.get("OPENAI_MODEL")
        or "(unset)"
    )
    return provider, model


def _new_session_id() -> str:
    """Generate a short, human-readable session id."""

    return uuid.uuid4().hex[:12]


def _print_header(
    out: TextIO,
    ctx: ChatContext,
    workspace_root: Path,
    *,
    resumed_from: str | None = None,
) -> None:
    """Render the AVO banner with version, model, session, and cwd.

    The header is a compact ASCII box so it survives every terminal
    width without word-wrap damage. Labels are fixed-width so the
    values line up. ``cwd`` is the resolved absolute path of the active
    workspace; ``session`` is the chat thread id (uuid-prefix); and
    ``model`` is whatever the runtime actually selected from the
    provider config.
    """

    cwd = Path.cwd()
    rows: list[tuple[str, str]] = [
        ("provider", f"{ctx.provider_name}"),
        ("model", f"{ctx.model_name}"),
        ("session", ctx.session_id),
        ("workspace", str(workspace_root)),
        ("cwd", str(cwd)),
        ("python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
    ]
    if resumed_from:
        rows.append(("resumed", resumed_from))

    label_width = max(len(label) for label, _ in rows)
    title_prefix = f" AVO v{AVO_VERSION} "
    max_value_width = max(len(value) for _, value in rows)
    inner_width = max(
        len(title_prefix) + 4,
        label_width + 3 + max_value_width,  # "label : value"
    )
    inner_width = max(inner_width, 40)
    inner_width = min(inner_width, 100)

    def _fit(value: str) -> str:
        budget = inner_width - label_width - 3
        if len(value) >= budget:
            return value[: max(budget - 3, 0)] + "..."
        return value.ljust(budget)

    title_dash_count = inner_width - len(title_prefix)
    out.write("\n")
    out.write(f"╭{title_prefix}{'─' * title_dash_count}╮\n")
    for label, value in rows:
        padded_label = label.ljust(label_width)
        out.write(f"│{padded_label} : {_fit(value)}│\n")
    out.write(f"╰{'─' * inner_width}╯\n")
    out.write(
        "Slash commands: /sessions, /resume [ID], /inspect RUN_ID, /skills, /skill NAME, "
        "/session, /new, /provider, /quit\n"
    )
    out.write("Enter a task to run one AgentRuntime turn. Ctrl+D or /quit to exit.\n")
    out.flush()


def _print_provider_summary(out: TextIO, ctx: ChatContext, environ: dict[str, str]) -> None:
    """Display provider/model without leaking API keys or tokens."""

    out.write(f"Provider: {ctx.provider_name}\n")
    out.write(f"Model: {ctx.model_name}\n")
    base_url = environ.get(f"AVO_{ctx.provider_name.upper()}_BASE_URL", "") or "(default)"
    out.write(f"Base URL: {base_url}\n")
    has_key = bool(environ.get(f"AVO_{ctx.provider_name.upper()}_API_KEY", "").strip())
    out.write(f"API key configured: {has_key}\n")
    out.flush()


def build_chat_context(
    *,
    database_path: Path,
    workspace_root: Path,
    environ: dict[str, str],
    session_id: str | None = None,
    force_new_session: bool = False,
) -> ChatContext:
    """Construct the runtime + store + workspace bound together.

    ``session_id`` optionally binds the chat thread to an existing
    session (used by ``/resume`` and the ``--session`` CLI flag). When
    ``force_new_session`` is true the chat always opens a fresh uuid
    thread even if ``session_id`` was provided. Raises
    :class:`AvoError` (or a subclass) on bad paths; ``ConfigError``
    propagates from :func:`build_provider_from_env`.
    """

    if database_path is None:
        raise AvoError("database_path must be a Path, not None; the CLI is misconfigured.")
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
    skills_root = workspace_root / ".avo" / "skills"
    # Ensure the skills directory exists for first-run use, but the
    # registry itself walks a path — body lookup happens lazily.
    skills_root.mkdir(parents=True, exist_ok=True)
    skills = SkillRegistry(skills_root)
    session = SessionLifecycle.open(db_path)
    if force_new_session or session_id is None:
        return ChatContext(
            runtime=runtime,
            store=store,
            workspace=workspace,
            provider_name=provider_name,
            model_name=model_name,
            skills=skills,
            session=session,
            session_id=_new_session_id(),
        )
    if not session.session_exists(session_id):
        session.close()
        raise AvoError(f"session {session_id!r} does not exist; nothing to resume.")
    preamble = session.build_preamble(session_id)
    return ChatContext(
        runtime=runtime,
        store=store,
        workspace=workspace,
        provider_name=provider_name,
        model_name=model_name,
        skills=skills,
        session=session,
        session_id=session_id,
        pending_preamble=preamble,
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

    if cmd == "/sessions":
        infos = ctx.session.list_sessions()
        out.write(render_session_picker(infos))
        return False

    if cmd == "/session":
        last = ctx.session.last_turn(ctx.session_id)
        turn_count = len(ctx.session.turns(ctx.session_id))
        out.write(f"Current session: {ctx.session_id}\n")
        out.write(f"Turns so far: {turn_count}\n")
        if last is not None:
            out.write(f"Last activity: {last.created_at.isoformat()}\n")
        return False

    if cmd == "/new":
        old = ctx.session_id
        ctx.session_id = _new_session_id()
        ctx.pending_preamble = None
        out.write(f"Closed session {old}; started fresh session {ctx.session_id}.\n")
        return False

    if cmd == "/inspect":
        if len(args) != 2:
            err.write("usage: /inspect RUN_ID\n")
            return False
        try:
            trace = await TraceInspector(ctx.store).inspect(args[1])
        except AvoError as exc:
            err.write(f"inspect failed: {exc}\n")
            return False
        out.write(trace.to_text())
        out.write("\n")
        return False

    if cmd == "/resume":
        # Two distinct resume shapes coexist here:
        #   /resume RUN_ID        -- resume a persisted runtime run (existing behaviour)
        #   /resume SESSION_ID    -- load a chat session thread for the next turn
        #   /resume (no args)     -- interactive picker over past chat sessions
        # /resume SESSION_ID wins over /resume RUN_ID when the arg
        # resolves to a known session — runtime-run ids never collide
        # with the short hex session ids we mint in ``_new_session_id``.
        if len(args) == 1:
            infos = ctx.session.list_sessions()
            if not infos:
                err.write("no previous chat sessions to resume.\n")
                return False
            out.write(render_session_picker(infos))
            return False
        if len(args) == 2:
            arg = args[1]
            infos = ctx.session.list_sessions()
            resolved = resolve_session_id(arg, infos)
            if resolved is not None:
                return await _resume_chat_session(ctx, resolved, out, err)
            # Fall back to runtime-run resume (existing behaviour).
            try:
                result = await ctx.runtime.resume(arg)
            except AvoError as exc:
                err.write(f"resume failed: {exc}\n")
                return False
            out.write(
                f"Resumed run {result.run_id}: status={result.status.value} "
                f"stop_reason={result.stop_reason.value} steps={result.steps}\n"
            )
            if result.output:
                out.write(f"output: {result.output}\n")
            return False
        err.write("usage: /resume [SESSION_ID|RUN_ID]\n")
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
        except AvoError as exc:
            err.write(f"skill load failed: {exc}\n")
            return False
        # Skill bodies are injected as a user message so the runtime
        # treats them like any other turn input — no separate channel.
        await _run_turn(ctx, body, out, err)
        return False

    err.write(f"unknown command: {cmd}\n")
    return False


async def _resume_chat_session(
    ctx: ChatContext,
    session_id: str,
    out: TextIO,
    err: TextIO,
) -> bool:
    """Switch the current thread to ``session_id`` and stage its preamble."""

    if session_id == ctx.session_id:
        out.write(f"Already on session {session_id}.\n")
        return False
    if not ctx.session.session_exists(session_id):
        err.write(f"session {session_id!r} does not exist.\n")
        return False
    try:
        preamble = ctx.session.build_preamble(session_id)
    except AvoError as exc:
        err.write(f"could not build resume preamble: {exc}\n")
        return False
    ctx.session_id = session_id
    ctx.pending_preamble = preamble
    turns = ctx.session.turns(session_id)
    out.write(
        f"Resumed session {session_id} with {len(turns)} prior turn(s). "
        "Next user message will be sent as a continuation.\n"
    )
    return False


async def _run_turn(ctx: ChatContext, task: str, out: TextIO, err: TextIO) -> None:
    """Execute one user turn against ``ctx.runtime``."""

    ctx.session.record_user_turn(ctx.session_id, task)
    effective_task = task
    if ctx.pending_preamble is not None:
        effective_task = (
            f"{ctx.pending_preamble}\n\n"
            f"---\n"
            f"User's current message (continue directly without greeting):\n{task}"
        )
        ctx.pending_preamble = None

    with bind_workspace(ctx.workspace):
        try:
            result = await ctx.runtime.run(effective_task)
        except AvoError as exc:
            err.write(f"runtime error: {exc}\n")
            return
        except Exception as exc:
            err.write(f"unexpected error: {type(exc).__name__}: {exc}\n")
            return

    assistant_content = result.output or ""
    ctx.session.record_assistant_turn(
        ctx.session_id,
        assistant_content,
        run_id=result.run_id,
        status=result.status.value,
        stop_reason=result.stop_reason.value,
    )

    out.write(
        f"Avo [{result.status.value}/{result.stop_reason.value}] "
        f"steps={result.steps} run_id={result.run_id}\n"
    )
    if result.error:
        out.write(f"error: {result.error}\n")
    if result.output:
        out.write(f"> {result.output}\n")
    out.flush()


def _maybe_offer_resume_prompt(
    session: SessionLifecycle,
    out: TextIO,
    in_stream: TextIO,
) -> SessionInfo | None:
    """If the last session is recent, offer to resume it. Return the chosen row.

    Used as a best-effort UX hook on REPL startup. Returns ``None``
    when there are no recent sessions or the operator declines. Does
    not block on EOF.
    """

    recent = session.find_resumable(limit=1)
    if not recent:
        return None
    info = recent[0]
    out.write(f"Resume session {info.session_id} from {render_session_row(info)}? [Y/n/custom] ")
    out.flush()
    try:
        answer = in_stream.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        out.write("\n")
        return None
    if answer in ("", "y", "yes"):
        return info
    return None


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
    session_id: str | None = None,
    force_new_session: bool = False,
) -> int:
    """Run the interactive chat REPL until EOF, /quit, or fatal init error.

    ``session_id`` and ``force_new_session`` mirror the ``--session``
    and ``--new-session`` CLI flags. When both are ``None``/``False``
    the REPL offers to resume the most-recent session before booting.
    """

    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout
    err_stream = stderr or sys.stderr
    env = environ if environ is not None else _read_environ()

    try:
        ctx = build_chat_context(
            database_path=database_path,
            workspace_root=workspace_root,
            environ=env,
            session_id=session_id,
            force_new_session=force_new_session,
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
                session_id=session_id,
                force_new_session=force_new_session,
            )
        except (AvoError, OSError) as exc:
            err_stream.write(f"avo chat: {exc}\n")
            return 2
        except ConfigError as exc:
            err_stream.write(f"configuration still invalid after setup: {exc}\n")
            return 2
    except (AvoError, OSError) as exc:
        err_stream.write(f"avo chat: {exc}\n")
        return 2

    resumed_from: str | None = None
    if session_id is not None and not force_new_session:
        resumed_from = session_id
    elif session_id is None and not force_new_session:
        offered = _maybe_offer_resume_prompt(ctx.session, out_stream, in_stream)
        if offered is not None:
            ctx.session_id = offered.session_id
            try:
                ctx.pending_preamble = ctx.session.build_preamble(offered.session_id)
            except AvoError:
                ctx.pending_preamble = None
            resumed_from = offered.session_id

    _print_header(out_stream, ctx, workspace_root, resumed_from=resumed_from)
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
        ctx.session.close()


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
