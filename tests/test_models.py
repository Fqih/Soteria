"""Domain-model validation and serialization edge cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from hernness.exceptions import InvalidStateTransitionError
from hernness.models import RunRecord, RunResult, TokenUsage, ToolResult, new_id
from hernness.state import RunState, StopReason


def test_generated_identifier_is_canonical_uuid_string() -> None:
    identifier = new_id()

    assert str(UUID(identifier)) == identifier


def test_datetime_values_are_normalized_to_utc() -> None:
    local = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=7)))
    run = RunRecord(task="timezone", created_at=local, updated_at=local)

    assert run.created_at.tzinfo is UTC
    assert run.created_at.hour == 17
    assert run.created_at.day == 31


@pytest.mark.parametrize(
    "values",
    [
        {"success": True, "error": "unexpected"},
        {"success": False},
        {
            "success": True,
            "started_at": datetime(2026, 1, 2, tzinfo=UTC),
            "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
        },
        {
            "success": True,
            "started_at": datetime(2026, 1, 1),
        },
    ],
)
def test_tool_result_rejects_inconsistent_payload(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ToolResult(tool_call_id="call", tool_name="tool", **values)


def test_run_record_rejects_stop_reason_before_terminal_state() -> None:
    with pytest.raises(ValidationError, match="non-terminal"):
        RunRecord(
            task="invalid",
            state=RunState.MODEL_PENDING,
            stop_reason=StopReason.MAX_STEPS,
        )


def test_run_record_rejects_reverse_timestamps() -> None:
    with pytest.raises(ValidationError, match="updated_at"):
        RunRecord(
            task="invalid",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_run_result_rejects_non_terminal_status() -> None:
    with pytest.raises(InvalidStateTransitionError):
        RunResult(
            run_id="run",
            status=RunState.MODEL_PENDING,
            stop_reason=StopReason.COMPLETED,
            steps=0,
            token_usage=TokenUsage(),
            token_accounting_available=True,
        )
