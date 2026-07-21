"""Trace normalization for error, policy, and lifecycle event variants."""

from __future__ import annotations

from soteria_loop.events import AgentEvent, EventType
from soteria_loop.models import RunRecord, TokenUsage
from soteria_loop.state import RunState, StopReason
from soteria_loop.tracing import RunTrace, TraceInspector


def test_summary_covers_all_operational_event_variants() -> None:
    cases = {
        EventType.MODEL_FAILED: ({"step": 2}, "provider failed"),
        EventType.TOOL_REQUESTED: ({"name": "x"}, "tool requested"),
        EventType.TOOL_APPROVAL_REQUESTED: ({"name": "x"}, "approval requested"),
        EventType.TOOL_APPROVED: ({"name": "x"}, "tool approved"),
        EventType.TOOL_DENIED: ({"name": "x"}, "tool denied"),
        EventType.TOOL_STARTED: ({"name": "x"}, "tool started"),
        EventType.TOOL_COMPLETED: ({"name": "x"}, "tool completed"),
        EventType.TOOL_FAILED: ({"name": "x"}, "tool failed"),
        EventType.POLICY_TRIGGERED: ({"policy": "max_steps"}, "policy stopped"),
        EventType.CHECKPOINT_CREATED: ({"state": "created"}, "checkpoint"),
        EventType.RUN_RESUMED: ({"checkpoint_id": "cp"}, "resumed"),
        EventType.RUN_FAILED: (
            {"state": "failed", "stop_reason": "internal_error"},
            "terminal",
        ),
        EventType.RUN_STOPPED: (
            {"state": "stopped", "stop_reason": "max_steps"},
            "terminal",
        ),
        EventType.RUN_CANCELLED: (
            {"state": "cancelled", "stop_reason": "user_cancelled"},
            "terminal",
        ),
    }

    for event_type, (payload, expected) in cases.items():
        event = AgentEvent(run_id="run", event_type=event_type, payload=payload)
        assert expected in TraceInspector._summary(event)


def test_trace_text_renders_duration_policy_and_error_suffixes() -> None:
    run = RunRecord.model_validate(
        {
            "run_id": "trace-run",
            "task": "trace",
            "state": RunState.STOPPED,
            "stop_reason": StopReason.MAX_STEPS,
            "duration_seconds": 1.25,
        }
    )
    events = [
        AgentEvent(
            run_id=run.run_id,
            sequence=1,
            event_type=EventType.POLICY_TRIGGERED,
            payload={
                "policy": "max_steps",
                "duration_ms": 2.5,
                "error": "detail",
            },
        )
    ]
    trace = TraceInspector.from_events(run, events)

    rendered = trace.to_text()
    assert "2.50ms" in rendered
    assert "policy=max_steps" in rendered
    assert "error=detail" in rendered


def test_non_policy_payload_is_not_rendered_as_policy_suffix() -> None:
    trace = RunTrace(
        run_id="run",
        entries=[],
        final_state=RunState.CREATED,
        stop_reason=None,
        steps=0,
        token_usage=TokenUsage(),
        token_accounting_available=True,
    )

    assert "Stop reason: -" in trace.to_text()
