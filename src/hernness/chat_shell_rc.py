"""Shell-rc persistence helpers for ``hernness chat``.

The operator's ``$SHELL`` decides whether ``~/.zshrc`` or ``~/.bashrc``
is written. The HERNNESS_* block is delimited by comment markers so
subsequent invocations replace the block in place rather than
accumulating duplicates, and the file is chmod 0o600 on POSIX.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TextIO

_SHELL_RC_MARKER_BEGIN = "# >>> soteria setup >>>"
_SHELL_RC_MARKER_END = "# <<< soteria setup <<<"


def _detect_shell_rc_path() -> Path | None:
    """Pick the shell rc file to update based on the current ``$SHELL``.

    Returns ``None`` when the shell is neither zsh nor bash, so the
    operator can opt-out by setting ``SHELL`` explicitly.
    """

    shell_path = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell_path:
        return home / ".zshrc"
    if "bash" in shell_path:
        return home / ".bashrc"
    return None


def _offer_persist_to_shell_rc(stdin: TextIO, stdout: TextIO, env: dict[str, str]) -> bool:
    """Ask the operator whether to write the config to their shell rc file.

    Returns ``True`` only when the operator types ``y`` or ``yes``
    (case-insensitive). Empty input, EOF, and anything else are NO.
    """

    rc_path = _detect_shell_rc_path()
    if rc_path is None:
        return False
    if not env:
        return False

    stdout.write(f"\nPersist these variables to {rc_path} so future shells see them? [y/N]: ")
    stdout.flush()
    line = stdin.readline()
    if not line:
        return False
    return line.strip().lower() in ("y", "yes")


def _quote_for_shell(value: str) -> str:
    """Single-quote-escape a value for POSIX shell double-quoted strings."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def persist_env_to_shell_rc(
    env: dict[str, str],
    *,
    rc_path: Path | None = None,
) -> Path:
    """Append (or replace) HERNNESS_* exports in ``rc_path``.

    The export block is delimited by ``# >>> soteria setup >>>`` and
    ``# <<< soteria setup <<<`` markers so subsequent invocations replace
    the block in place rather than accumulating duplicates. The block
    is written with POSIX-sh-compatible double-quoted exports so values
    containing single quotes survive intact.

    Returns the path that was written. Raises ``OSError`` if the path is
    not writable.
    """

    if rc_path is None:
        rc_path = _detect_shell_rc_path()
        if rc_path is None:
            raise OSError("Could not detect a shell rc file: $SHELL is neither zsh nor bash.")

    existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""

    # Drop the previous soteria block if present.
    start = existing.find(_SHELL_RC_MARKER_BEGIN)
    end = existing.find(_SHELL_RC_MARKER_END)
    if start != -1 and end != -1 and end > start:
        existing = existing[:start] + existing[end + len(_SHELL_RC_MARKER_END) :]

    # Only HERNNESS_* keys are persisted.
    lines = [f"{_SHELL_RC_MARKER_BEGIN}"]
    for key in sorted(env):
        if not key.startswith("HERNNESS_"):
            continue
        value = _quote_for_shell(env[key])
        lines.append(f'export {key}="{value}"')
    lines.append(_SHELL_RC_MARKER_END)
    block = "\n".join(lines) + "\n"

    # Append with a leading blank line for readability unless the file
    # was empty or already ended with one.
    suffix = block
    if existing and not existing.endswith("\n\n"):
        suffix = ("\n" if not existing.endswith("\n") else "") + block

    rc_path.write_text(existing + suffix, encoding="utf-8")
    # Best-effort restrictive perms on POSIX. We don't fail the run if
    # chmod does not work (e.g. Windows or non-POSIX filesystem).
    with contextlib.suppress(OSError):
        rc_path.chmod(0o600)

    return rc_path


__all__ = [
    "_SHELL_RC_MARKER_BEGIN",
    "_SHELL_RC_MARKER_END",
    "_detect_shell_rc_path",
    "_offer_persist_to_shell_rc",
    "_quote_for_shell",
    "persist_env_to_shell_rc",
]
