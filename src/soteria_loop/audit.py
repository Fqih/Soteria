"""Append-only structured JSONL audit log.

The runtime emits one JSON object per audit event. Events capture
security-sensitive decisions (tool calls, approval outcomes, provider
calls) for offline review. Secrets are redacted before write — keys
matching common API-key patterns get their value replaced with
``"[redacted]"`` so a leaked log file never exfiltrates credentials.

Log lines are flushed and fsync'd on close so a crash mid-run never
leaves a half-written tail.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from soteria_loop.exceptions import SoteriaError

AuditError = SoteriaError

_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "auth",
        "authorization",
        "secret",
        "password",
        "passwd",
        "credential",
        "credentials",
    }
)
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xoxb-[A-Za-z0-9-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _REDACT_VALUE.sub("[redacted]", value)
    if isinstance(value, Mapping):
        return {
            str(k): ("[redacted]" if _is_secret_key(str(k)) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return lowered in _SECRET_KEYS or lowered.endswith("_token") or lowered.endswith("_key")


_REDACT_VALUE = re.compile("|".join(f"(?:{p.pattern})" for p in _SECRET_VALUE_PATTERNS))


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
            "event": self.event,
            "timestamp": self.timestamp.isoformat(),
            "payload": _redact(dict(self.payload)),
        }
        if self.run_id is not None:
            out["run_id"] = self.run_id
        if self.session_id is not None:
            out["session_id"] = self.session_id
        if self.actor is not None:
            out["actor"] = self.actor
        return out


class AuditLog:
    """Append-only JSONL sink."""

    __slots__ = ("_closed", "_path")

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def write(self, event: AuditEvent) -> None:
        if self._closed:
            raise AuditError(f"audit log at {self._path} is closed")
        line = json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True)
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

    return cast(dict[str, Any], _redact(dict(payload)))


__all__ = ["AuditError", "AuditEvent", "AuditLog", "redact"]
