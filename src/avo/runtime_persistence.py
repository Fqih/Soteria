"""Persistence helpers for :mod:`avo.runtime`.

Extracted from ``runtime.py`` so the public class stays below the
file-size limit. These free functions take ``(runtime, context, ...)``
and call :meth:`AgentRuntime._event_persisted` after each durable
write so failure-injection tests still observe every event.

The functions are private to the runtime module; callers outside the
runtime should not import them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic import JsonValue, TypeAdapter

from avo.events import AgentEvent, EventType
from avo.exceptions import UnsafeResumeError
from avo.models import (
    Checkpoint,
    RunRecord,
    ToolResult,
)
from avo.providers.base import StatefulModelProvider
from avo.state import RunState, StopReason, is_terminal, validate_transition

if TYPE_CHECKING:
    from avo.runtime import AgentRuntime, _RunContext

_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


async def append(
    runtime: AgentRuntime,
    context: _RunContext,
    event_type: EventType,
    payload: dict[str, JsonValue],
) -> AgentEvent:
    event = AgentEvent(
        run_id=context.run.run_id,
        event_type=event_type,
        created_at=runtime._now(),
        payload=payload,
    )
    persisted = await runtime.event_store.append_event(event)
    await runtime._event_persisted(persisted)
    return persisted


async def append_with_run(
    runtime: AgentRuntime,
    context: _RunContext,
    event_type: EventType,
    payload: dict[str, JsonValue],
) -> AgentEvent:
    event = AgentEvent(
        run_id=context.run.run_id,
        event_type=event_type,
        created_at=runtime._now(),
        payload=payload,
    )
    persisted = await runtime.event_store.append_event_and_update_run(event, context.run)
    await runtime._event_persisted(persisted)
    return persisted


async def transition(
    runtime: AgentRuntime,
    context: _RunContext,
    state: RunState,
) -> AgentEvent:
    validate_transition(context.run.state, state)
    updated = runtime._active_record(context, state=state)
    event = AgentEvent(
        run_id=context.run.run_id,
        event_type=EventType.STATE_CHANGED,
        created_at=updated.updated_at,
        payload={
            "from_state": context.run.state.value,
            "to_state": state.value,
        },
    )
    persisted = await runtime.event_store.append_event_and_update_run(event, updated)
    context.run = updated
    await runtime._event_persisted(persisted)
    return persisted


async def checkpoint(
    runtime: AgentRuntime,
    context: _RunContext,
) -> Checkpoint:
    snapshot = Checkpoint(
        run_id=context.run.run_id,
        created_at=runtime._now(),
        state=context.run.state,
        messages=context.messages,
        next_step=context.next_step,
        token_usage=context.token_usage,
        token_accounting_available=context.token_accounting_available,
        consecutive_errors=context.consecutive_errors,
        repeated_action_history=context.detector.action_history,
        observation_fingerprints=context.detector.observation_history,
        model_response_fingerprints=context.detector.model_history,
        progress_markers=context.detector.progress_markers,
        completed_tool_call_ids=context.completed_tool_call_ids,
        user_state=context.user_state,
        policy=cast(dict[str, JsonValue], context.policy.model_dump(mode="json")),
        provider_metadata=provider_snapshot(runtime),
        pending_response=context.pending_response,
    )
    event = AgentEvent(
        run_id=context.run.run_id,
        event_type=EventType.CHECKPOINT_CREATED,
        created_at=snapshot.created_at,
        payload={
            "checkpoint_id": snapshot.checkpoint_id,
            "state": snapshot.state.value,
            "next_step": snapshot.next_step,
        },
    )
    persisted_checkpoint, persisted_event = await runtime.event_store.save_checkpoint(
        snapshot, event
    )
    await runtime._event_persisted(persisted_event)
    return persisted_checkpoint


async def reconcile_after_checkpoint(
    runtime: AgentRuntime,
    context: _RunContext,
    checkpoint_obj: Checkpoint,
) -> None:
    events = await runtime.event_store.get_events(context.run.run_id)
    trailing = [event for event in events if event.sequence > checkpoint_obj.last_event_sequence]
    started_ids = {
        str(event.payload["tool_call_id"])
        for event in trailing
        if event.event_type is EventType.TOOL_STARTED
        and isinstance(event.payload.get("tool_call_id"), str)
    }
    result_events = [
        event
        for event in trailing
        if event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
    ]
    result_ids: set[str] = set()
    for event in result_events:
        raw = event.payload.get("result")
        if not isinstance(raw, dict):
            raise UnsafeResumeError(
                f"Tool result event {event.event_id!r} has no reconstructable result payload."
            )
        result = ToolResult.model_validate(raw)
        result_ids.add(result.tool_call_id)
        if result.tool_call_id not in context.completed_tool_call_ids:
            record_tool_result(runtime, context, result)
            if result.success:
                context.consecutive_errors = 0
            else:
                context.consecutive_errors += 1
            context.pending_response = None

    uncertain = started_ids - result_ids - context.completed_tool_call_ids
    if uncertain:
        call_ids = ", ".join(sorted(uncertain))
        raise UnsafeResumeError(
            "Cannot safely resume because these tool calls started without a durable "
            f"result: {call_ids}. Inspect their external side effects before retrying."
        )

    if result_ids and context.run.state is RunState.TOOL_EXECUTING:
        context.run = runtime._active_record(context)
        await transition(runtime, context, RunState.OBSERVATION_RECORDED)
        await checkpoint(runtime, context)


def record_tool_result(
    runtime: AgentRuntime,
    context: _RunContext,
    result: ToolResult,
) -> None:
    context.completed_tool_call_ids.add(result.tool_call_id)
    context.detector.record_observation(result)
    context.messages.append(
        {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "name": result.tool_name,
            "content": cast(JsonValue, result.model_dump(mode="json")),
        }
    )


def provider_snapshot(runtime: AgentRuntime) -> dict[str, JsonValue]:
    if isinstance(runtime.provider, StatefulModelProvider):
        return _JSON_OBJECT_ADAPTER.validate_python(runtime.provider.snapshot_state())
    return {}


def restore_provider(runtime: AgentRuntime, metadata: dict[str, JsonValue]) -> None:
    if not metadata:
        return
    if not isinstance(runtime.provider, StatefulModelProvider):
        raise UnsafeResumeError(
            "The checkpoint contains provider state, but the configured provider "
            "does not implement restore_state()."
        )
    runtime.provider.restore_state(metadata)


async def trigger_policy(
    runtime: AgentRuntime,
    context: _RunContext,
    reason: StopReason,
) -> None:
    await append(
        runtime,
        context,
        EventType.POLICY_TRIGGERED,
        {
            "policy": reason.value,
            "stop_reason": reason.value,
            "state": context.run.state.value,
            "steps": context.run.steps,
            "elapsed_seconds": runtime._elapsed(context),
        },
    )
    await terminate(runtime, context, RunState.STOPPED, reason)


async def terminate(
    runtime: AgentRuntime,
    context: _RunContext,
    state: RunState,
    reason: StopReason,
    *,
    output: str | None = None,
    error: str | None = None,
) -> None:
    if is_terminal(context.run.state):
        return
    validate_transition(context.run.state, state)

    # Preserve a resumable pre-terminal boundary, even when per-step snapshots are off.
    await checkpoint(runtime, context)
    now = runtime._now()
    terminal_record = RunRecord.model_validate(
        {
            **context.run.model_dump(),
            "state": state,
            "stop_reason": reason,
            "output": output,
            "error": error,
            "steps": context.run.steps,
            "token_usage": context.token_usage,
            "token_accounting_available": context.token_accounting_available,
            "user_state": context.user_state,
            "updated_at": now,
            "duration_seconds": runtime._elapsed(context, now=now),
        }
    )
    state_event = AgentEvent(
        run_id=context.run.run_id,
        event_type=EventType.STATE_CHANGED,
        created_at=now,
        payload={
            "from_state": context.run.state.value,
            "to_state": state.value,
        },
    )
    terminal_type = {
        RunState.COMPLETED: EventType.RUN_COMPLETED,
        RunState.FAILED: EventType.RUN_FAILED,
        RunState.STOPPED: EventType.RUN_STOPPED,
        RunState.CANCELLED: EventType.RUN_CANCELLED,
    }[state]
    terminal_event = AgentEvent(
        run_id=context.run.run_id,
        event_type=terminal_type,
        created_at=now,
        payload={
            "state": state.value,
            "stop_reason": reason.value,
            "output": output,
            "error": error,
            "steps": context.run.steps,
            "token_usage": cast(JsonValue, context.token_usage.model_dump(mode="json")),
            "token_accounting_available": context.token_accounting_available,
        },
    )
    persisted_state, persisted_terminal = await runtime.event_store.finalize_run(
        terminal_record,
        state_event,
        terminal_event,
    )
    context.run = terminal_record
    await runtime._event_persisted(persisted_state)
    await runtime._event_persisted(persisted_terminal)


__all__ = [
    "append",
    "append_with_run",
    "checkpoint",
    "provider_snapshot",
    "reconcile_after_checkpoint",
    "record_tool_result",
    "restore_provider",
    "terminate",
    "transition",
    "trigger_policy",
]


# Re-export the type-checker alias for runtime.py imports.
__annotations__ = {"_RunContext": "_RunContext"}  # silence linters on unused alias
