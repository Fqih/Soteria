"""Shell-rc persistence helpers for ``avo chat``.

The operator's ``$SHELL`` decides whether ``~/.zshrc`` or ``~/.bashrc``
is written. The AVO_* block is delimited by comment markers so
subsequent invocations replace the block in place rather than
accumulating duplicates, and the file is chmod 0o600 on POSIX.

Writes are atomic: the new file content is rendered to a sibling
``*.avo.tmp`` first, then ``os.replace`` swaps it onto the target path.
A crash mid-write leaves the previous rc untouched.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TextIO

_SHELL_RC_MARKER_BEGIN = "# >>> avo setup >>>"
_SHELL_RC_MARKER_END = "# <<< avo setup <<<"
_BOM = "﻿"


def _strip_bom(text: str) -> str:
    """Strip a leading UTF-8 BOM if present (preserve the rest verbatim)."""

    return text[1:] if text.startswith(_BOM) else text


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
    """Escape a value for POSIX-shell double-quoted strings.

    Inside double quotes, the shell still expands ``$``, backticks, and
    history characters (in interactive zsh). Newlines split the export
    into invalid syntax. Escape all of those so a stray API key with a
    newline or ``$`` round-trips intact.
    """

    # Order matters: backslashes first so we don't double-escape later
    # replacements.
    out = value.replace("\\", "\\\\")
    out = out.replace('"', '\\"')
    out = out.replace("$", "\\$")
    out = out.replace("`", "\\`")
    out = out.replace("\n", "\\n")
    out = out.replace("\r", "\\r")
    return out


def _atomic_write_text(target: Path, content: str) -> None:
    """Write ``content`` to ``target`` via a sibling temp + ``os.replace``.

    The temp file sits next to ``target`` so the rename stays on the
    same filesystem (atomic on POSIX and on Windows since 3.3).
    """

    tmp = target.with_name(target.name + ".avo.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        # If the rename succeeded the temp is gone; otherwise clean up
        # so we don't leak partial writes.
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def persist_env_to_shell_rc(
    env: dict[str, str],
    *,
    rc_path: Path | None = None,
) -> Path:
    """Append (or replace) AVO_* exports in ``rc_path``.

    The export block is delimited by ``# >>> avo setup >>>`` and
    ``# <<< avo setup <<<`` markers so subsequent invocations replace
    the block in place rather than accumulating duplicates. The block
    is written with POSIX-sh-compatible double-quoted exports so values
    containing single quotes, dollar signs, backticks, or embedded
    newlines survive intact. Writes are atomic — a crash mid-write
    leaves the previous rc untouched.

    Returns the path that was written. Raises ``OSError`` if the path is
    not writable.
    """

    if rc_path is None:
        rc_path = _detect_shell_rc_path()
        if rc_path is None:
            raise OSError("Could not detect a shell rc file: $SHELL is neither zsh nor bash.")

    existing = _strip_bom(rc_path.read_text(encoding="utf-8")) if rc_path.exists() else ""

    # Drop the previous avo block if present.
    start = existing.find(_SHELL_RC_MARKER_BEGIN)
    end = existing.find(_SHELL_RC_MARKER_END)
    if start != -1 and end != -1 and end > start:
        existing = existing[:start] + existing[end + len(_SHELL_RC_MARKER_END) :]

    # Only AVO_* keys are persisted.
    lines = [_SHELL_RC_MARKER_BEGIN]
    for key in sorted(env):
        if not key.startswith("AVO_"):
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

    _atomic_write_text(rc_path, existing + suffix)
    # Best-effort restrictive perms on POSIX. We don't fail the run if
    # chmod does not work (e.g. Windows or non-POSIX filesystem).
    with contextlib.suppress(OSError):
        rc_path.chmod(0o600)

    return rc_path


__all__ = [
    "_SHELL_RC_MARKER_BEGIN",
    "_SHELL_RC_MARKER_END",
    "_atomic_write_text",
    "_detect_shell_rc_path",
    "_offer_persist_to_shell_rc",
    "_quote_for_shell",
    "_strip_bom",
    "persist_env_to_shell_rc",
]
