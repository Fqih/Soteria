"""Tests for the audit log."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from avo.audit import AuditError, AuditEvent, AuditLog, redact


def test_redact_replaces_known_secret_keys() -> None:
    payload = {"api_key": "sk-test", "name": "alice"}
    out = redact(payload)
    assert out["api_key"] == "[redacted]"
    assert out["name"] == "alice"


def test_redact_rejects_all_suffix_variants() -> None:
    payload = {
        "openai_api_key": "x",
        "openrouter_token": "x",
        "client_secret": "x",
        "db_password": "x",
        "auth_header": "x",
        "proxy_authorization": "x",
        "admin_passwd": "x",
        "user_credential": "x",
        "service_credentials": "x",
        "data": "ok",
    }
    out = redact(payload)
    assert out["data"] == "ok"
    for secret_key in (
        "openai_api_key",
        "openrouter_token",
        "client_secret",
        "db_password",
        "auth_header",
        "proxy_authorization",
        "admin_passwd",
        "user_credential",
        "service_credentials",
    ):
        assert out[secret_key] == "[redacted]"


def test_redact_does_not_match_unrelated_words() -> None:
    payload = {"donkey": "value", "hockey": "score", "username": "alice"}
    out = redact(payload)
    assert out["donkey"] == "value"
    assert out["hockey"] == "score"
    assert out["username"] == "alice"


def test_redact_matches_secret_values_in_strings() -> None:
    payload = {"message": "key=sk-abcdefghijklmnopqrstuv ok"}
    out = redact(payload)
    assert "sk-abcdef" not in out["message"]
    assert "[redacted]" in out["message"]


def test_redact_matches_jwt_aws_gitlab_patterns() -> None:
    payload = {
        "a": "Bearer eyJabcdefghij.eyJabcdefghij.eyJabcdefghij",
        "b": "AKIAIOSFODNN7EXAMPLE",
        "c": "glpat-abcdefghijklmnopqrst",
        "d": "xoxp-1234567890-1234567890",
        "e": "sk_live_abcdefghijklmnopqrst",
        "f": "password=hunter2hunter",
    }
    out = redact(payload)
    assert "eyJ" not in out["a"]
    assert "AKIAIOSFODNN7EXAMPLE" not in out["b"]
    assert "glpat-" not in out["c"]
    assert "xoxp-" not in out["d"]
    assert "sk_live_" not in out["e"]
    assert "hunter2hunter" not in out["f"]


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


def test_redact_handles_bytes() -> None:
    out = redact({"bin": b"secret-bytes"})
    assert out["bin"] == "[redacted]"


def test_redact_handles_set_and_frozenset() -> None:
    out = redact({"tags": {"alpha", "beta"}, "frozen": frozenset({"x"})})
    assert isinstance(out["tags"], list)
    assert sorted(out["tags"]) == ["alpha", "beta"]
    assert isinstance(out["frozen"], list)


def test_redact_handles_cyclic_payload() -> None:
    payload: dict[str, object] = {"name": "root"}
    payload["self"] = payload
    out = redact(payload)
    assert out["name"] == "root"
    assert out["self"] == "[redacted:cycle]"


def test_redact_handles_shared_subtree() -> None:
    sub = {"api_key": "x"}
    payload = {"a": sub, "b": sub}
    out = redact(payload)
    assert out["a"]["api_key"] == "[redacted]"
    assert out["b"]["api_key"] == "[redacted]"


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


def test_audit_event_redacts_metadata_fields() -> None:
    event = AuditEvent(
        event="login",
        actor="admin_password=hunter2hunter2",
        run_id="abc",
        session_id="xyz",
    )
    out = event.to_dict()
    assert "[redacted]" in out["actor"]
    assert "hunter2hunter2" not in out["actor"]


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


def test_audit_log_run_and_session_optional() -> None:
    event = AuditEvent(event="x")
    out = event.to_dict()
    assert "run_id" not in out
    assert "session_id" not in out
    assert "actor" not in out


def test_audit_log_writes_non_serializable_payload_as_audit_error(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    class NotSerializable:
        def __repr__(self) -> str:
            raise RuntimeError("repr exploded")

        def __str__(self) -> str:
            raise RuntimeError("str exploded")

    with pytest.raises(AuditError, match="not JSON-serializable"):
        log.write(AuditEvent(event="x", payload={"obj": NotSerializable()}))


def test_audit_log_writes_datetime_payload_via_default(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    log.write(AuditEvent(event="x", payload={"at": datetime.now(UTC)}))
    log.close()
    data = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert "at" in data["payload"]


def test_audit_log_serializes_uuid_payload(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    token = uuid4()
    log.write(AuditEvent(event="x", payload={"id": token}))
    log.close()
    line = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert str(token) in line


def test_audit_log_refuses_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "real.jsonl"
    target.write_text("")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(AuditError, match="symlink"):
        AuditLog(link)


def test_audit_log_concurrent_writes_do_not_interleave(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    errors: list[Exception] = []

    def writer(prefix: str) -> None:
        try:
            for i in range(50):
                log.write(AuditEvent(event=f"{prefix}-{i}"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"t{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.close()
    assert errors == []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    for line in lines:
        json.loads(line)  # every line parseable → no interleaving
