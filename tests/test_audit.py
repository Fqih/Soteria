"""Tests for the audit log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soteria_loop.audit import AuditError, AuditEvent, AuditLog, redact


def test_redact_replaces_known_secret_keys() -> None:
    payload = {"api_key": "sk-test", "name": "alice"}
    out = redact(payload)
    assert out["api_key"] == "[redacted]"
    assert out["name"] == "alice"


def test_redact_replaces_suffix_secret_keys() -> None:
    payload = {"openai_api_key": "abc", "openrouter_token": "def", "data": "ok"}
    out = redact(payload)
    assert out["openai_api_key"] == "[redacted]"
    assert out["openrouter_token"] == "[redacted]"
    assert out["data"] == "ok"


def test_redact_matches_secret_values_in_strings() -> None:
    payload = {"message": "key=sk-abcdefghijklmnopqrstuv ok"}
    out = redact(payload)
    assert "sk-abcdef" not in out["message"]
    assert "[redacted]" in out["message"]


def test_redact_nested_mapping_and_list() -> None:
    payload = {"outer": {"token": "abc", "ok": 1}, "items": [{"secret": "x"}, {"ok": 2}]}
    out = redact(payload)
    assert out["outer"]["token"] == "[redacted]"
    assert out["outer"]["ok"] == 1
    assert out["items"][0]["secret"] == "[redacted]"
    assert out["items"][1]["ok"] == 2


def test_redact_returns_non_string_scalars() -> None:
    assert redact({"n": 5}) == {"n": 5}
    assert redact({"b": True}) == {"b": True}
    assert redact({"none": None}) == {"none": None}


def test_audit_event_to_dict_basic() -> None:
    event = AuditEvent(event="tool.call", payload={"name": "read_file"}, run_id="r1")
    out = event.to_dict()
    assert out["event"] == "tool.call"
    assert out["run_id"] == "r1"
    assert out["payload"] == {"name": "read_file"}
    assert "timestamp" in out


def test_audit_event_redacts_payload() -> None:
    event = AuditEvent(event="provider.call", payload={"api_key": "sk-abc"})
    out = event.to_dict()
    assert out["payload"]["api_key"] == "[redacted]"


def test_audit_log_writes_one_json_per_line(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    log.write(AuditEvent(event="a", payload={"k": 1}))
    log.write(AuditEvent(event="b", payload={"k": 2}))
    log.close()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "a"
    assert json.loads(lines[1])["event"] == "b"


def test_audit_log_write_many(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    log.write_many(
        [
            AuditEvent(event="a"),
            AuditEvent(event="b"),
            AuditEvent(event="c"),
        ]
    )
    log.close()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_audit_log_rejects_write_after_close(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.close()
    with pytest.raises(AuditError, match="closed"):
        log.write(AuditEvent(event="x"))


def test_audit_log_context_manager(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    with AuditLog(log_path) as log:
        log.write(AuditEvent(event="x"))
    assert log_path.exists()


def test_audit_log_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "audit.jsonl"
    log = AuditLog(nested)
    log.write(AuditEvent(event="x"))
    log.close()
    assert nested.exists()


def test_audit_event_run_and_session_optional() -> None:
    event = AuditEvent(event="x")
    out = event.to_dict()
    assert "run_id" not in out
    assert "session_id" not in out
    assert "actor" not in out
