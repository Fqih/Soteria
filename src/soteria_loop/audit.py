"""Append-only structured JSONL audit log.

The runtime emits one JSON object per audit event. Events capture
security-sensitive decisions (tool calls, approval outcomes, provider
calls) for offline review. Secrets are redacted before write — keys
matching common API-key patterns get their value replaced with
``"[redacted]"`` so a leaked log file never exfiltrates credentials.

Log lines are fsync'd after each write so a crash mid-run never leaves
a half-written tail. Writes are serialized via a process-wide lock so
concurrent appenders never interleave bytes within a line.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from soteria_loop.exceptions import SoteriaError

AuditError = SoteriaError

_SECRET_SUFFIXES: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "auth",
    "authorization",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "key",
)

_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"xoxb-[A-Za-z0-9-]{20,}"),
    re.compile(r"xoxp-[A-Za-z0-9-]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk_test_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?:password|secret|token)=([^\s&;]{8,})", re.IGNORECASE),
)

_REDACT_VALUE = re.compile(
    "|".join(f"(?:{p.pattern})" for p in _VALUE_PATTERNS),
    re.IGNORECASE,
)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    tokens = lowered.split("_")
    return any(token in _SECRET_SUFFIXES for token in tokens)


def _redact(value: Any, _seen: set[int] | None = None) -> Any:
    if isinstance(value, str):
        return _REDACT_VALUE.sub("[redacted]", value)
    if isinstance(value, (bytes, bytearray)):
        return "[redacted]"
    if isinstance(value, (set, frozenset)):
        items = sorted((_redact(v, _seen) for v in value), key=repr)
        return items
    if isinstance(value, Mapping):
        seen = _seen if _seen is not None else set()
        if id(value) in seen:
            return "[redacted:cycle]"
        seen = seen | {id(value)}
        return {
            str(k): ("[redacted]" if _is_secret_key(str(k)) else _redact(v, seen))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        seen = _seen if _seen is not None else set()
        if id(value) in seen:
            return "[redacted:cycle]"
        seen = seen | {id(value)}
        return [_redact(v, seen) for v in value]
    return value


@dataclass(frozen=True)
class AuditEvent:
    """A single record written to the log."""

    event: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None = None
    session_id: str | None = None
    actor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event": _redact(self.event) if isinstance(self.event, str) else self.event,
            "timestamp": self.timestamp.isoformat(),
            "payload": cast(dict[str, Any], _redact(dict(self.payload))),
        }
        if self.run_id is not None:
            out["run_id"] = _redact(self.run_id)
        if self.session_id is not None:
            out["session_id"] = _redact(self.session_id)
        if self.actor is not None:
            out["actor"] = _redact(self.actor)
        return out


class AuditLog:
    """Append-only JSONL sink."""

    __slots__ = ("_closed", "_lock", "_path")

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AuditError(f"could not create audit log directory: {exc}") from exc
        if self._path.exists() and self._path.is_symlink():
            raise AuditError(f"audit log path {self._path} is a symlink; refusing to follow")
        self._lock = threading.Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def write(self, event: AuditEvent) -> None:
        if self._closed:
            raise AuditError(f"audit log at {self._path} is closed")
        try:
            line = json.dumps(
                event.to_dict(),
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
        except Exception as exc:
            raise AuditError(f"audit event not JSON-serializable: {exc}") from exc
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.write("\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError as exc:
                raise AuditError(f"failed to write audit event: {exc}") from exc

    def write_many(self, events: Iterable[AuditEvent]) -> None:
        for event in events:
            self.write(event)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def redact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Public redaction helper — used by callers building their own payloads."""

    return cast(dict[str, Any], _redact(payload))


__all__ = ["AuditError", "AuditEvent", "AuditLog", "redact"]
