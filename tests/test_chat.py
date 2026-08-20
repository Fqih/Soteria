"""Tests for the ``soteria-loop chat`` REPL and its dependencies."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from soteria_loop import ModelResponse
from soteria_loop.app_tools.file_tools import bind_workspace, read_file_tool, write_file_tool
from soteria_loop.chat import build_chat_context, run_repl
from soteria_loop.config import ConfigError
from soteria_loop.exceptions import SoteriaError
from soteria_loop.providers.base import ModelProvider
from soteria_loop.storage.sqlite import SQLiteEventStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _ScriptedProvider(ModelProvider):
    """Provider that returns pre-recorded responses and records calls."""

    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = list(script)
        self.calls: list[str] = []

    async def generate(self, request):  # type: ignore[override]
        self.calls.append(request.messages[-1].get("content", ""))  # type: ignore[union-attr]
        if not self.script:
            from soteria_loop.exceptions import FakeProviderExhaustedError

            raise FakeProviderExhaustedError("script empty")
        return self.script.pop(0)


@pytest.fixture
def chat_env(tmp_path: Path) -> dict[str, Path]:
    db = tmp_path / "chat.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("seed", encoding="utf-8")
    return {"db": db, "workspace": workspace}


def _environ_with_ollama(model: str = "fake-test-model") -> dict[str, str]:
    """Return an env dict that ``build_provider_from_env`` will accept for ollama."""

    return {
        "SOTERIA_PROVIDER": "ollama",
        "SOTERIA_MODEL": model,
        "SOTERIA_OLLAMA_BASE_URL": "http://example.invalid",
    }


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_cli_chat_subcommand_parses(chat_env: dict[str, Path]) -> None:
    """The ``chat`` subcommand accepts an optional ``--workspace-root``."""

    from soteria_loop.cli import _parser

    parser = _parser()
    args = parser.parse_args(
        [
            "chat",
            "--database",
            str(chat_env["db"]),
            "--workspace-root",
            str(chat_env["workspace"]),
        ]
    )
    assert args.command == "chat"
    assert args.workspace_root == chat_env["workspace"]
    assert args.database == chat_env["db"]


def test_cli_existing_runs_subcommands_still_parse(chat_env: dict[str, Path]) -> None:
    """Existing ``runs list/inspect/resume`` continue to work after the chat add."""

    from soteria_loop.cli import _parser

    parser = _parser()
    list_args = parser.parse_args(["runs", "list"])
    assert list_args.command == "runs"
    assert list_args.runs_command == "list"

    inspect_args = parser.parse_args(["runs", "inspect", "abc"])
    assert inspect_args.runs_command == "inspect"
    assert inspect_args.run_id == "abc"

    resume_args = parser.parse_args(["runs", "resume", "def"])
    assert resume_args.runs_command == "resume"
    assert resume_args.run_id == "def"


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def test_build_chat_context_constructs_runtime_with_tools(
    chat_env: dict[str, Path],
) -> None:
    """Context wires the provider factory, SQLite store, and file tools."""

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=_environ_with_ollama(),
    )

    assert isinstance(ctx.runtime.provider, ModelProvider)
    assert ctx.store is not None
    assert ctx.workspace.root == chat_env["workspace"].resolve()
    tool_names = {tool.metadata.name for tool in ctx.runtime.tools._tools.values()}  # type: ignore[attr-defined]
    assert tool_names == {"read_file", "write_file"}


def test_build_chat_context_missing_provider_raises(
    chat_env: dict[str, Path],
) -> None:
    with pytest.raises(ConfigError):
        build_chat_context(
            database_path=chat_env["db"],
            workspace_root=chat_env["workspace"],
            environ={},
        )


def test_build_chat_context_missing_workspace_raises(
    chat_env: dict[str, Path],
) -> None:
    with pytest.raises(SoteriaError):
        build_chat_context(
            database_path=chat_env["db"],
            workspace_root=chat_env["workspace"] / "does-not-exist",
            environ=_environ_with_ollama(),
        )


# ---------------------------------------------------------------------------
# REPL delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repl_runs_one_turn_per_non_empty_line(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each user line becomes one ``AgentRuntime.run`` invocation."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    scripted = _ScriptedProvider([ModelResponse(content="hi"), ModelResponse(content="hey")])
    ctx.runtime.provider = scripted

    stdout = io.StringIO()
    stderr = io.StringIO()

    # Inline REPL using the same machinery but with the scripted provider swapped in.
    from soteria_loop.chat import _run_turn

    with bind_workspace(ctx.workspace):
        for line in ("hello", "world"):
            await _run_turn(ctx, line, stdout, stderr)

    assert scripted.calls == ["hello", "world"]


@pytest.mark.asyncio
async def test_repl_quit_command_exits(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/quit`` and ``/exit`` return exit code 0 and close the store."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    stdin = io.StringIO("/quit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=env,
    )

    assert code == 0
    assert "Slash commands" in stdout.getvalue()


@pytest.mark.asyncio
async def test_repl_exit_alias_exits(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environ_with_ollama()
    stdin = io.StringIO("/exit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=env,
    )
    assert code == 0


@pytest.mark.asyncio
async def test_repl_provider_command_does_not_leak_key(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/provider`` shows model + key presence but never the key value."""

    env = {
        "SOTERIA_PROVIDER": "ollama",
        "SOTERIA_MODEL": "llama3.1",
        "SOTERIA_OLLAMA_BASE_URL": "http://localhost:11434",
        "SOTERIA_OLLAMA_API_KEY": "super-secret-token",
    }
    monkeypatch.setattr("os.environ", env)

    stdin = io.StringIO("/provider\n/quit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=env,
    )
    assert code == 0
    output = stdout.getvalue()
    assert "super-secret-token" not in output
    assert "Provider: ollama" in output
    assert "Model: llama3.1" in output
    assert "API key configured: True" in output


@pytest.mark.asyncio
async def test_repl_empty_lines_are_skipped(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    stdin = io.StringIO("\n\n   \n/quit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=env,
    )
    assert code == 0


@pytest.mark.asyncio
async def test_repl_eof_exits_cleanly(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    stdin = io.StringIO("")  # immediate EOF
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=env,
    )
    assert code == 0
    assert "Slash commands" in stdout.getvalue()


@pytest.mark.asyncio
async def test_repl_runtime_error_does_not_crash(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime error inside one turn is contained; the REPL continues."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    scripted = _ScriptedProvider([])  # raises FakeProviderExhaustedError on first call
    ctx.runtime.provider = scripted

    from soteria_loop.chat import _run_turn

    stdout = io.StringIO()
    stderr = io.StringIO()
    with bind_workspace(ctx.workspace):
        # First turn: provider raises, runtime catches as PROVIDER_ERROR,
        # result has status=FAILED. _run_turn renders stop_reason + error.
        await _run_turn(ctx, "trigger error", stdout, stderr)
        # Second turn after the error: verify containment rather than crash.
        await _run_turn(ctx, "trigger again", stdout, stderr)

    output = stdout.getvalue() + stderr.getvalue()
    assert "failed" in output.lower() or "provider" in output.lower()


# ---------------------------------------------------------------------------
# Workspace protections survive the chat path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_workspace_blocks_traversal(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``validate_path`` inside the chat context still rejects escapes."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    with pytest.raises(Exception, match="escapes"):
        ctx.workspace.validate_path("../escape.txt")


@pytest.mark.asyncio
async def test_chat_file_tools_work_inside_workspace(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read_file_tool and write_file_tool resolve against the chat workspace."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )

    with bind_workspace(ctx.workspace):
        result = await write_file_tool().invoke({"path": "via_chat.txt", "content": "written"})
        assert result["size"] == 7

        read_result = await read_file_tool().invoke({"path": "via_chat.txt"})
        assert read_result["content"] == "written"

    assert (chat_env["workspace"] / "via_chat.txt").read_text(encoding="utf-8") == "written"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_persists_run_and_events(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    scripted = _ScriptedProvider([ModelResponse(content="done")])
    ctx.runtime.provider = scripted

    stdout = io.StringIO()
    stderr = io.StringIO()

    from soteria_loop.chat import _run_turn

    with bind_workspace(ctx.workspace):
        await _run_turn(ctx, "first task", stdout, stderr)

    await ctx.store.close()

    # Re-open with a fresh store to confirm the row is durable.
    store2 = SQLiteEventStore(chat_env["db"])
    runs = await store2.list_runs()
    assert len(runs) == 1
    run = runs[0]
    events = await store2.get_events(run.run_id)
    # RUN_CREATED + MODEL_REQUESTED + MODEL_RESPONDED + STATE_CHANGED + RUN_FINALIZED
    assert len(events) >= 4
    await store2.close()


@pytest.mark.asyncio
async def test_chat_multiple_turns_create_multiple_runs(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    scripted = _ScriptedProvider(
        [ModelResponse(content="a"), ModelResponse(content="b"), ModelResponse(content="c")]
    )
    ctx.runtime.provider = scripted

    from soteria_loop.chat import _run_turn

    stdout = io.StringIO()
    stderr = io.StringIO()
    with bind_workspace(ctx.workspace):
        for task in ("one", "two", "three"):
            await _run_turn(ctx, task, stdout, stderr)

    await ctx.store.close()
    store2 = SQLiteEventStore(chat_env["db"])
    runs = await store2.list_runs()
    assert len(runs) == 3
    await store2.close()


@pytest.mark.asyncio
async def test_chat_inspect_command_finds_persisted_run(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    scripted = _ScriptedProvider([ModelResponse(content="ok")])
    ctx.runtime.provider = scripted

    stdout = io.StringIO()
    stderr = io.StringIO()

    from soteria_loop.chat import _run_turn

    with bind_workspace(ctx.workspace):
        await _run_turn(ctx, "task one", stdout, stderr)

    run_id = (await ctx.store.list_runs())[0].run_id

    # Now invoke /inspect through the slash dispatcher.
    from soteria_loop.chat import _run_slash

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_requested = await _run_slash(ctx, ["/inspect", run_id], stdout, stderr, env)
    assert exit_requested is False
    assert run_id in stdout.getvalue()


# ---------------------------------------------------------------------------
# CLI integration smoke (no provider network calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_full_flow_until_quit(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: header printed, turn executed, ``/quit`` exits 0."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    ctx = build_chat_context(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        environ=env,
    )
    scripted = _ScriptedProvider([ModelResponse(content="hello")])
    ctx.runtime.provider = scripted

    # Patch run_repl's internal ctx by injecting ours via a thin wrapper.
    from soteria_loop import chat as chat_module

    original_build = chat_module.build_chat_context
    chat_module.build_chat_context = lambda **kwargs: ctx  # type: ignore[assignment]
    try:
        stdin = io.StringIO("hello there\n/quit\n")
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = await run_repl(
            database_path=chat_env["db"],
            workspace_root=chat_env["workspace"],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            environ=env,
        )
    finally:
        chat_module.build_chat_context = original_build  # type: ignore[assignment]

    assert code == 0
    output = stdout.getvalue()
    assert "Provider: ollama" in output
    assert "Workspace:" in output
    assert scripted.calls == ["hello there"]


# ---------------------------------------------------------------------------
# First-run configuration UX
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repl_missing_provider_starts_interactive_setup(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SOTERIA_PROVIDER -> interactive setup wizard. Empty stdin aborts cleanly."""

    monkeypatch.setattr("os.environ", {})

    stdin = io.StringIO()
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ={},
    )

    assert code == 2
    # Wizard banner shown on stdout.
    out = stdout.getvalue()
    assert "Soteria First-Time Setup" in out
    assert "Select your provider" in out
    assert "1. Ollama" in out
    # Aborted on EOF.
    assert "setup aborted" in out.lower() or "aborted" in out.lower()


@pytest.mark.asyncio
async def test_repl_invalid_provider_starts_interactive_setup(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown provider still triggers the interactive setup."""

    monkeypatch.setattr(
        "os.environ",
        {"SOTERIA_PROVIDER": "anthropic-clone", "SOTERIA_MODEL": "x"},
    )

    # Provide "1\n" then EOF to abort the rest cleanly (Ollama needs no key).
    stdin = io.StringIO("")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ={"SOTERIA_PROVIDER": "anthropic-clone", "SOTERIA_MODEL": "x"},
    )

    assert code == 2
    assert "Soteria First-Time Setup" in stdout.getvalue()


@pytest.mark.asyncio
async def test_repl_valid_provider_still_enters_chat(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid SOTERIA_PROVIDER config still drives the REPL normally."""

    env = _environ_with_ollama()
    monkeypatch.setattr("os.environ", env)

    stdin = io.StringIO("/quit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ=env,
    )

    assert code == 0
    # Header printed, not the setup wizard.
    assert "Soteria First-Time Setup" not in stdout.getvalue()
    assert "Slash commands" in stdout.getvalue()


@pytest.mark.asyncio
async def test_repl_first_run_does_not_leak_secrets(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run wizard text never echoes an API key value."""

    secret = "super-secret-token-should-never-appear"
    monkeypatch.setattr("os.environ", {"SOTERIA_PROVIDER": secret, "SOTERIA_MODEL": "x"})

    stdin = io.StringIO("")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ={"SOTERIA_PROVIDER": secret, "SOTERIA_MODEL": "x"},
    )

    assert code == 2
    combined = stdout.getvalue() + stderr.getvalue()
    assert secret not in combined
    # Common API-key prefixes must never appear in the friendly message.
    for prefix in ("sk-", "sk-ant-", "Bearer "):
        assert prefix not in combined


def test_existing_runs_subcommands_unchanged(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``runs list/inspect/resume`` still parse and run unchanged."""

    from soteria_loop.cli import main

    db = chat_env["db"]
    code = main(["--database", str(db), "runs", "list"])
    assert code == 0

    code = main(["--database", str(db), "runs", "inspect", "nope"])
    assert code == 2  # missing run, but parser + dispatcher ok


# ---------------------------------------------------------------------------
# Interactive first-run setup unit tests
# ---------------------------------------------------------------------------


def test_setup_ollama_completes() -> None:
    from soteria_loop.chat import interactive_first_run_setup

    # 1 = ollama (no API key, default model)
    stdin = io.StringIO("1\n\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "x")
    assert env == {
        "SOTERIA_PROVIDER": "ollama",
        "SOTERIA_MODEL": "llama3.1",
    }
    assert "Soteria First-Time Setup" in stdout.getvalue()
    assert "Starting Soteria" in stdout.getvalue()


def test_setup_openai_with_key() -> None:
    from soteria_loop.chat import interactive_first_run_setup

    # 2 = openai, then API key (from secret_reader), empty base url, empty model
    stdin = io.StringIO("2\n\n\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "sk-test123")
    assert env == {
        "SOTERIA_PROVIDER": "openai",
        "SOTERIA_OPENAI_API_KEY": "sk-test123",
        "SOTERIA_MODEL": "gpt-5.6",
    }


def test_setup_anthropic_uses_custom_model() -> None:
    from soteria_loop.chat import interactive_first_run_setup

    # 3 = anthropic, key (secret_reader), model override
    stdin = io.StringIO("3\nclaude-opus\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "sk-ant-test")
    assert env == {
        "SOTERIA_PROVIDER": "anthropic",
        "SOTERIA_ANTHROPIC_API_KEY": "sk-ant-test",
        "SOTERIA_MODEL": "claude-opus",
    }


def test_setup_minimax_with_api_style() -> None:
    from soteria_loop.chat import interactive_first_run_setup

    # 4 = minimax, key (secret_reader), style=openai, default model
    stdin = io.StringIO("4\nopenai\n\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "key-xyz")
    assert env == {
        "SOTERIA_PROVIDER": "minimax",
        "SOTERIA_MINIMAX_API_KEY": "key-xyz",
        "SOTERIA_MINIMAX_API_STYLE": "openai",
        "SOTERIA_MODEL": "MiniMax-M3",
    }


def test_setup_invalid_choice_re_prompts() -> None:
    from soteria_loop.chat import interactive_first_run_setup

    # 9 invalid, 0 invalid, then 1 valid
    stdin = io.StringIO("9\n0\n1\n\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "x")
    assert env is not None
    assert env["SOTERIA_PROVIDER"] == "ollama"
    out = stdout.getvalue()
    assert "please choose one of" in out


def test_setup_eof_returns_none() -> None:
    from soteria_loop.chat import interactive_first_run_setup

    stdin = io.StringIO("")  # immediate EOF
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "x")
    assert env is None
    assert "aborted" in stdout.getvalue().lower()


def test_setup_uses_secret_reader_not_stdin() -> None:
    """API key must come from secret_reader, not from the REPL stdin."""

    from soteria_loop.chat import interactive_first_run_setup

    secret = "shhh-do-not-leak"

    def fake_secret_reader(prompt: str) -> str:
        return secret

    stdin = io.StringIO("2\n\n\n")  # newlines after each prompt
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=fake_secret_reader)
    assert env is not None
    assert env["SOTERIA_OPENAI_API_KEY"] == secret
    # The secret must NOT appear in stdout (it was typed via secret_reader).
    assert secret not in stdout.getvalue()


@pytest.mark.asyncio
async def test_repl_setup_succeeds_then_quits(
    chat_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After interactive setup completes, the REPL starts and accepts /quit."""

    monkeypatch.setattr("os.environ", {})

    # 1 (ollama, no API key), empty line for default model, /quit
    stdin = io.StringIO("1\n\n/quit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=chat_env["db"],
        workspace_root=chat_env["workspace"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ={},
        secret_reader=lambda _: "irrelevant",  # Ollama doesn't read a key
    )

    assert code == 0
    out = stdout.getvalue()
    assert "Soteria First-Time Setup" in out
    assert "Provider configured: Ollama" in out
    assert "Slash commands" in out  # REPL header after setup
