"""Tests for shell-rc persistence: opt-in prompt and idempotent rewrite.

Covers ``persist_env_to_shell_rc`` (block delimiting, replacement of
previous block, preservation of user lines, quoting of special chars,
refusal when the shell is neither zsh nor bash) and
``_offer_persist_to_shell_rc`` (default NO, accept y/yes, abort on EOF,
skip when env empty). End-to-end setup -> prompt -> write tests are
also here.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shell rc persistence
# ---------------------------------------------------------------------------


def test_persist_appends_avo_block_to_new_rc(tmp_path: Path) -> None:
    """Empty rc file: persist creates the file with a marked block."""

    from avo.chat import persist_env_to_shell_rc

    rc = tmp_path / ".zshrc"
    env = {
        "AVO_PROVIDER": "minimax",
        "AVO_MINIMAX_API_KEY": "secret-value",
        "AVO_MODEL": "MiniMax-M3",
        "PATH": "/usr/bin",  # non-AVO_ keys must NOT be persisted
    }
    written = persist_env_to_shell_rc(env, rc_path=rc)

    assert written == rc
    content = rc.read_text(encoding="utf-8")
    assert "# >>> avo setup >>>" in content
    assert "# <<< avo setup <<<" in content
    assert 'export AVO_PROVIDER="minimax"' in content
    assert 'export AVO_MINIMAX_API_KEY="secret-value"' in content
    assert 'export AVO_MODEL="MiniMax-M3"' in content
    # non-AVO_ vars never reach the rc file.
    assert "PATH" not in content


def test_persist_replaces_previous_block(tmp_path: Path) -> None:
    """A second invocation overwrites the previous block, no duplicates."""

    from avo.chat import persist_env_to_shell_rc

    rc = tmp_path / ".zshrc"
    persist_env_to_shell_rc(
        {"AVO_PROVIDER": "ollama", "AVO_OLLAMA_API_KEY": "first"},
        rc_path=rc,
    )
    persist_env_to_shell_rc(
        {"AVO_PROVIDER": "openai", "AVO_OPENAI_API_KEY": "second"},
        rc_path=rc,
    )

    content = rc.read_text(encoding="utf-8")
    assert content.count("# >>> avo setup >>>") == 1
    assert content.count("# <<< avo setup <<<") == 1
    assert "first" not in content
    assert "second" in content
    assert "ollama" not in content.split("# >>> avo setup >>>")[0]


def test_persist_preserves_user_lines_above_and_below(tmp_path: Path) -> None:
    """User content outside the avo block must survive a rewrite."""

    from avo.chat import persist_env_to_shell_rc

    rc = tmp_path / ".bashrc"
    rc.write_text(
        "alias ll='ls -la'\nexport EDITOR=vim\n",
        encoding="utf-8",
    )
    persist_env_to_shell_rc(
        {"AVO_PROVIDER": "openai", "AVO_OPENAI_API_KEY": "k"},
        rc_path=rc,
    )

    content = rc.read_text(encoding="utf-8")
    assert "alias ll='ls -la'" in content
    assert "export EDITOR=vim" in content
    assert "AVO_OPENAI_API_KEY" in content


def test_persist_quotes_special_characters(tmp_path: Path) -> None:
    """Keys with single quotes / backslashes survive shell re-source."""

    from avo.chat import persist_env_to_shell_rc

    rc = tmp_path / ".zshrc"
    persist_env_to_shell_rc(
        {"AVO_OPENAI_API_KEY": "a'b\"c\\d"},
        rc_path=rc,
    )
    content = rc.read_text(encoding="utf-8")
    # Backslash before quote, backslash before backslash.
    assert (
        'export AVO_OPENAI_API_KEY="a\\\'b\\"c\\\\d"' in content
        or 'export AVO_OPENAI_API_KEY="a\'b\\"c\\\\d"' in content
    )


def test_persist_returns_path_written(tmp_path: Path) -> None:
    from avo.chat import persist_env_to_shell_rc

    rc = tmp_path / ".bashrc"
    result = persist_env_to_shell_rc(
        {"AVO_PROVIDER": "anthropic", "AVO_ANTHROPIC_API_KEY": "k"},
        rc_path=rc,
    )
    assert result == rc


def test_persist_refuses_to_write_when_shell_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from avo.chat import persist_env_to_shell_rc

    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    with pytest.raises(OSError, match="shell rc"):
        persist_env_to_shell_rc({"AVO_PROVIDER": "ollama"})


def test_offer_persist_defaults_no() -> None:
    from avo.chat import _offer_persist_to_shell_rc

    env = {"AVO_PROVIDER": "ollama", "AVO_OLLAMA_API_KEY": "k"}

    stdin = io.StringIO("\n")  # default empty
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, env) is False

    stdin = io.StringIO("n\n")
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, env) is False

    stdin = io.StringIO("no\n")
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, env) is False


def test_offer_persist_accepts_y_and_yes() -> None:
    from avo.chat import _offer_persist_to_shell_rc

    env = {"AVO_PROVIDER": "ollama", "AVO_OLLAMA_API_KEY": "k"}

    stdin = io.StringIO("y\n")
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, env) is True

    stdin = io.StringIO("YES\n")
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, env) is True


def test_offer_persist_aborts_on_eof() -> None:
    from avo.chat import _offer_persist_to_shell_rc

    env = {"AVO_PROVIDER": "ollama", "AVO_OLLAMA_API_KEY": "k"}

    stdin = io.StringIO("")  # EOF
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, env) is False


def test_offer_persist_skips_when_env_empty() -> None:
    from avo.chat import _offer_persist_to_shell_rc

    stdin = io.StringIO("y\n")
    stdout = io.StringIO()
    assert _offer_persist_to_shell_rc(stdin, stdout, {}) is False


def test_setup_prompts_to_persist_then_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: setup completes, prompt is shown, accept writes rc."""

    from avo.chat import interactive_first_run_setup

    rc = tmp_path / ".zshrc"
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr("avo.chat.Path.home", lambda: tmp_path)

    stdin = io.StringIO("4\ny\n2\n\n\ny\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "shhh-secret")

    assert env is not None
    # y must have been consumed by the persist prompt, so the next read
    # for api_style would have gotten "2".
    assert env["AVO_MINIMAX_API_STYLE"] == "openai"
    assert env["AVO_MINIMAX_API_KEY"] == "shhh-secret"

    # The rc file got the AVO_ block.
    assert rc.exists()
    content = rc.read_text(encoding="utf-8")
    assert 'AVO_MINIMAX_API_KEY="shhh-secret"' in content


def test_setup_default_decline_does_not_write_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup completes; default NO leaves the rc file untouched."""

    from avo.chat import interactive_first_run_setup

    rc = tmp_path / ".zshrc"
    rc.write_text("alias ll='ls -la'\n", encoding="utf-8")
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr("avo.chat.Path.home", lambda: tmp_path)

    stdin = io.StringIO("1\n\n\n\n")
    stdout = io.StringIO()
    env = interactive_first_run_setup(stdin, stdout, secret_reader=lambda _: "k")

    assert env is not None
    # The rc file is untouched (no avo block).
    content = rc.read_text(encoding="utf-8")
    assert "avo setup" not in content
    assert "alias ll='ls -la'" in content
