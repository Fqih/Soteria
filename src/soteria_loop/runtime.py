"""Explicit async state-machine runtime for bounded agent loops."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from soteria_loop.events import AgentEvent, EventType
from soteria_loop.exceptions import (
    CheckpointNotFoundError,
    FakeProviderExhaustedError,
    ProviderError,
    RunAlreadyTerminalError,
    StorageError,
    ToolAlreadyCompletedError,
    UnsafeResumeError,
)
from soteria_loop.integrations.lethe import LetheMemoryAdapter
from soteria_loop.models import (
    Checkpoint,
    ModelRequest,
    ModelResponse,
    RunRecord,
    RunResult,
    TokenUsage,
    ToolCall,
    ToolResult,
    utc_now,
)
from soteria_loop.policies import LoopPolicy
from soteria_loop.progress import ProgressDetector
from soteria_loop.providers.base import ModelProvider, StatefulModelProvider
from soteria_loop.state import RunState, StopReason, is_terminal, validate_transition
from soteria_loop.storage.base import EventStore
from soteria_loop.storage.memory import InMemoryEventStore
from soteria_loop.tools import Tool, ToolRegistry, tool_call_fingerprint

if TYPE_CHECKING:
    from soteria_loop.tracing import RunTrace

Clock = Callable[[], datetime]
ApprovalCallback = Callable[[ToolCall], bool | Awaitable[bool]]
_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


async def _always_approve(call: ToolCall) -> bool:
    """Exercise the approval state in v0.1 without implementing approval policy."""

    del call
    return True


@dataclass
class _RunContext:
    run: RunRecord
    policy: LoopPolicy
    messages: list[dict[str, JsonValue]]
    next_step: int
    token_usage: TokenUsage
    token_accounting_available: bool
    consecutive_errors: int
    detector: ProgressDetector
    completed_tool_call_ids: set[str]
    user_state: dict[str, JsonValue]
    pending_response: ModelResponse | None = None


class AgentRuntime:
    """Coordinate model decisions, tools, policies, events, and checkpoints.

    One runtime instance serializes its own runs because a provider may maintain
    cursor state. Different runtime instances can operate independently.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        tools: Iterable[Tool] = (),
        policy: LoopPolicy | None = None,
        event_store: EventStore | None = None,
        clock: Clock = utc_now,
        approval_callback: ApprovalCallback | None = None,
        memory: LetheMemoryAdapter | None = None,
    ) -> None:
        self.provider = provider
        self.tools = ToolRegistry(tools)
        self.policy = policy or LoopPolicy()
        self.event_store = event_store or InMemoryEventStore()
        self._clock = clock
        self._approval_callback = approval_callback or _always_approve
        self.memory = memory
        self._execution_lock = asyncio.Lock()

    async def run(
        self,
        task: str,
        *,
        user_state: dict[str, JsonValue] | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        """Create and execute a run until it reaches one explicit terminal state."""

        async with self._execution_lock:
            now = self._now()
            values: dict[str, object] = {
                "task": task,
                "user_state": user_state or {},
                "created_at": now,
                "updated_at": now,
            }
            if run_id is not None:
                values["run_id"] = run_id
            record = RunRecord.model_validate(values)
            messages: list[dict[str, JsonValue]] = [{"role": "user", "content": task}]
            if self.memory is not None:
                recalled = self.memory.recall_text(task)
                if recalled:
                    messages.append(
                        {
                            "role": "system",
                            "content": "Relevant memories:\n- " + "\n- ".join(recalled),
                        }
                    )
            context = _RunContext(
                run=record,
                policy=self.policy,
                messages=messages,
                next_step=1,
                token_usage=TokenUsage(),
                token_accounting_available=True,
                consecutive_errors=0,
                detector=ProgressDetector(),
                completed_tool_call_ids=set(),
                user_state=dict(user_state or {}),
            )
            created = AgentEvent(
                run_id=record.run_id,
                event_type=EventType.RUN_CREATED,
                created_at=now,
                payload={
                    "task": task,
                    "policy": cast(dict[str, JsonValue], self.policy.model_dump(mode="json")),
                },
            )
            persisted = await self.event_store.create_run(record, created)
            await self._event_persisted(persisted)

            try:
                await self._transition(context, RunState.MODEL_PENDING)
                await self._checkpoint(context)
                return await self._drive(context)
            except asyncio.CancelledError:
                if not is_terminal(context.run.state):
                    await self._terminate(
                        context,
                        RunState.CANCELLED,
                        StopReason.USER_CANCELLED,
                        error="Run cancelled by the caller.",
                    )
                raise
            except StorageError:
                raise
            except Exception as exc:
                if not is_terminal(context.run.state):
                    return await self._terminate(
                        context,
                        RunState.FAILED,
                        StopReason.INTERNAL_ERROR,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                return self._result(context.run)

    async def resume(self, run_id: str) -> RunResult:
        """Resume a persisted non-terminal run from its latest safe checkpoint."""

        async with self._execution_lock:
            record = await self.event_store.get_run(run_id)
            if is_terminal(record.state):
                raise RunAlreadyTerminalError(
                    f"Run {run_id!r} is already terminal ({record.state.value}) and cannot resume."
                )
            checkpoint = await self.event_store.get_latest_checkpoint(run_id)
            if checkpoint is None:
                raise CheckpointNotFoundError(
                    f"Run {run_id!r} has no checkpoint. Resume requires at least one "
                    "successfully persisted checkpoint."
                )

            policy = LoopPolicy.model_validate(checkpoint.policy)
            context = _RunContext(
                run=record,
                policy=policy,
                messages=[dict(message) for message in checkpoint.messages],
                next_step=checkpoint.next_step,
                token_usage=checkpoint.token_usage.model_copy(),
                token_accounting_available=checkpoint.token_accounting_available,
                consecutive_errors=checkpoint.consecutive_errors,
                detector=ProgressDetector(
                    action_history=checkpoint.repeated_action_history,
                    observation_history=checkpoint.observation_fingerprints,
                    model_history=checkpoint.model_response_fingerprints,
                    progress_markers=checkpoint.progress_markers,
                ),
                completed_tool_call_ids=set(checkpoint.completed_tool_call_ids),
                user_state=dict(checkpoint.user_state),
                pending_response=checkpoint.pending_response,
            )
            self._restore_provider(checkpoint.provider_metadata)
            await self._reconcile_after_checkpoint(context, checkpoint)

            resumed = await self._append(
                context,
                EventType.RUN_RESUMED,
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_sequence": checkpoint.last_event_sequence,
                    "state": context.run.state.value,
                },
            )
            del resumed
            try:
                return await self._drive(context)
            except asyncio.CancelledError:
                if not is_terminal(context.run.state):
                    await self._terminate(
                        context,
                        RunState.CANCELLED,
                        StopReason.USER_CANCELLED,
                        error="Resumed run cancelled by the caller.",
                    )
                raise
            except StorageError:
                raise
            except Exception as exc:
                if not is_terminal(context.run.state):
                    return await self._terminate(
                        context,
                        RunState.FAILED,
                        StopReason.INTERNAL_ERROR,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                return self._result(context.run)

    async def inspect(self, run_id: str) -> RunTrace:
        """Build a chronological trace for a stored run."""

        from soteria_loop.tracing import TraceInspector

        return await TraceInspector(self.event_store).inspect(run_id)

    async def _drive(self, context: _RunContext) -> RunResult:
        handlers = {
            RunState.CREATED: self._handle_created,
            RunState.MODEL_PENDING: self._handle_model_pending,
            RunState.DECISION_RECEIVED: self._handle_decision_received,
            RunState.TOOL_PENDING: self._handle_tool_pending,
            RunState.APPROVAL_PENDING: self._handle_approval_pending,
            RunState.TOOL_EXECUTING: self._handle_tool_executing,
            RunState.OBSERVATION_RECORDED: self._handle_observation_recorded,
            RunState.PAUSED: self._handle_paused,
        }
        while not is_terminal(context.run.state):
            handler = handlers.get(context.run.state)
            if handler is None:
                raise RuntimeError(f"No state handler exists for {context.run.state.value!r}.")
            await handler(context)
        return self._result(context.run)

    async def _handle_created(self, context: _RunContext) -> None:
        await self._transition(context, RunState.MODEL_PENDING)

    async def _handle_model_pending(self, context: _RunContext) -> None:
        boundary_reason = self._operation_boundary_reason(context)
        if boundary_reason is not None:
            await self._trigger_policy(context, boundary_reason)
            return
        if context.next_step > context.policy.max_steps:
            await self._trigger_policy(context, StopReason.MAX_STEPS)
            return

        step = context.next_step
        context.next_step += 1
        request = ModelRequest(
            run_id=context.run.run_id,
            step=step,
            messages=context.messages,
            tools=self.tools.metadata,
        )
        context.run = self._active_record(context, steps=step)
        await self._append_with_run(
            context,
            EventType.MODEL_REQUESTED,
            {
                "step": step,
                "request": cast(JsonValue, request.model_dump(mode="json")),
            },
        )

        started = time.perf_counter()
        try:
            provider_call = self.provider.generate(request)
            if context.policy.provider_timeout_seconds is None:
                generated = await provider_call
            else:
                generated = await asyncio.wait_for(
                    provider_call,
                    timeout=context.policy.provider_timeout_seconds,
                )
            response = ModelResponse.model_validate(generated)
        except ValidationError as exc:
            await self._append(
                context,
                EventType.MODEL_FAILED,
                {
                    "step": step,
                    "error": str(exc),
                    "error_type": "invalid_model_response",
                    "duration_ms": (time.perf_counter() - started) * 1000,
                },
            )
            await self._terminate(
                context,
                RunState.FAILED,
                StopReason.INVALID_MODEL_RESPONSE,
                error=f"Provider returned an invalid model response: {exc}",
            )
            return
        except TimeoutError:
            timeout = context.policy.provider_timeout_seconds
            await self._handle_provider_error(
                context,
                step,
                started,
                ProviderError(
                    f"Provider call exceeded the configured timeout of {timeout} seconds.",
                    retryable=True,
                ),
            )
            return
        except Exception as exc:
            await self._handle_provider_error(context, step, started, exc)
            return

        if response.usage is None:
            context.token_accounting_available = False
        else:
            context.token_usage = context.token_usage.plus(response.usage)
        context.consecutive_errors = 0
        context.pending_response = response
        context.detector.record_model_response(response)
        if response.tool_call is not None:
            context.detector.record_action(response.tool_call)
        context.messages.append(self._assistant_message(response))
        context.run = self._active_record(context, error=None)

        await self._append(
            context,
            EventType.MODEL_RESPONDED,
            {
                "step": step,
                "response": cast(JsonValue, response.model_dump(mode="json")),
                "duration_ms": (time.perf_counter() - started) * 1000,
                "token_accounting_available": context.token_accounting_available,
            },
        )
        await self._transition(context, RunState.DECISION_RECEIVED)

        if context.policy.checkpoint_every_step or response.tool_call is not None:
            await self._checkpoint(context)

        token_reason = context.policy.token_budget_reason(
            context.token_usage,
            accounting_available=context.token_accounting_available,
        )
        if token_reason is not None:
            await self._trigger_policy(context, token_reason)

    async def _handle_provider_error(
        self,
        context: _RunContext,
        step: int,
        started: float,
        exc: Exception,
    ) -> None:
        context.consecutive_errors += 1
        context.run = self._active_record(
            context,
            error=f"{type(exc).__name__}: {exc}",
        )
        await self._append_with_run(
            context,
            EventType.MODEL_FAILED,
            {
                "step": step,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "consecutive_errors": context.consecutive_errors,
            },
        )
        await self._checkpoint(context)

        if isinstance(exc, FakeProviderExhaustedError) or (
            isinstance(exc, ProviderError) and not exc.retryable
        ):
            await self._terminate(
                context,
                RunState.FAILED,
                StopReason.PROVIDER_ERROR,
                error=f"Provider failed without a retry path: {exc}",
            )
        elif context.consecutive_errors >= context.policy.consecutive_error_limit:
            await self._trigger_policy(context, StopReason.CONSECUTIVE_ERRORS)

    async def _handle_decision_received(self, context: _RunContext) -> None:
        response = self._require_pending_response(context)
        if response.is_final:
            if self.memory is not None and response.content is not None:
                self.memory.remember_output(response.content, session_id=context.run.run_id)
            await self._terminate(
                context,
                RunState.COMPLETED,
                StopReason.COMPLETED,
                output=response.content,
            )
            return

        call = self._require_tool_call(response)
        await self._append(
            context,
            EventType.TOOL_REQUESTED,
            {
                "tool_call_id": call.tool_call_id,
                "idempotency_key": tool_call_fingerprint(call),
                "name": call.name,
                "arguments": call.arguments,
            },
        )
        if context.detector.repeated_action(context.policy.repeated_action_limit):
            await self._trigger_policy(context, StopReason.REPEATED_ACTION)
            return
        await self._transition(context, RunState.TOOL_PENDING)

    async def _handle_tool_pending(self, context: _RunContext) -> None:
        call = self._require_tool_call(self._require_pending_response(context))
        await self._append(
            context,
            EventType.TOOL_APPROVAL_REQUESTED,
            {
                "tool_call_id": call.tool_call_id,
                "idempotency_key": tool_call_fingerprint(call),
                "name": call.name,
                "mode": "v0.1_callback",
            },
        )
        await self._transition(context, RunState.APPROVAL_PENDING)

    async def _handle_approval_pending(self, context: _RunContext) -> None:
        call = self._require_tool_call(self._require_pending_response(context))
        approved_value = self._approval_callback(call)
        approved = await approved_value if inspect.isawaitable(approved_value) else approved_value
        if not approved:
            await self._append(
                context,
                EventType.TOOL_DENIED,
                {"tool_call_id": call.tool_call_id, "name": call.name},
            )
            await self._trigger_policy(context, StopReason.POLICY_DENIED)
            return
        await self._append(
            context,
            EventType.TOOL_APPROVED,
            {
                "tool_call_id": call.tool_call_id,
                "idempotency_key": tool_call_fingerprint(call),
                "name": call.name,
                "mode": "v0.1_callback",
            },
        )
        await self._transition(context, RunState.TOOL_EXECUTING)

    async def _handle_tool_executing(self, context: _RunContext) -> None:
        boundary_reason = self._operation_boundary_reason(context)
        if boundary_reason is not None:
            await self._trigger_policy(context, boundary_reason)
            return
        response = self._require_pending_response(context)
        call = self._require_tool_call(response)
        if call.tool_call_id in context.completed_tool_call_ids:
            raise ToolAlreadyCompletedError(
                f"Refusing to execute completed tool call {call.tool_call_id!r}."
            )

        await self._append(
            context,
            EventType.TOOL_STARTED,
            {
                "tool_call_id": call.tool_call_id,
                "idempotency_key": tool_call_fingerprint(call),
                "name": call.name,
                "arguments": call.arguments,
            },
        )
        started_at = utc_now()
        started_clock = time.perf_counter()
        try:
            invocation = self.tools.invoke(
                call,
                completed_tool_call_ids=context.completed_tool_call_ids,
            )
            if context.policy.tool_timeout_seconds is None:
                result = await invocation
            else:
                result = await asyncio.wait_for(
                    invocation,
                    timeout=context.policy.tool_timeout_seconds,
                )
        except TimeoutError:
            finished_at = utc_now()
            result = ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                success=False,
                error=(
                    f"Tool {call.name!r} exceeded the configured timeout of "
                    f"{context.policy.tool_timeout_seconds} seconds."
                ),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=max(0.0, (time.perf_counter() - started_clock) * 1000),
            )
        result_type = EventType.TOOL_COMPLETED if result.success else EventType.TOOL_FAILED
        await self._append(
            context,
            result_type,
            {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "result": cast(JsonValue, result.model_dump(mode="json")),
                "duration_ms": result.duration_ms,
                "error": result.error,
            },
        )

        self._record_tool_result(context, result)
        context.pending_response = None
        if result.success:
            context.consecutive_errors = 0
            context.run = self._active_record(context, error=None)
        else:
            context.consecutive_errors += 1
            context.run = self._active_record(context, error=result.error)
        await self._transition(context, RunState.OBSERVATION_RECORDED)

        # A successful or durably recorded tool result always forces a checkpoint.
        await self._checkpoint(context)

        boundary_reason = self._operation_boundary_reason(context)
        if boundary_reason is not None:
            await self._trigger_policy(context, boundary_reason)
        elif context.consecutive_errors >= context.policy.consecutive_error_limit:
            await self._trigger_policy(context, StopReason.CONSECUTIVE_ERRORS)
        elif context.detector.no_progress(context.policy.no_progress_window):
            await self._trigger_policy(context, StopReason.NO_PROGRESS)

    async def _handle_observation_recorded(self, context: _RunContext) -> None:
        await self._transition(context, RunState.MODEL_PENDING)

    async def _handle_paused(self, context: _RunContext) -> None:
        raise UnsafeResumeError(
            f"Run {context.run.run_id!r} is paused without a recorded continuation state."
        )

    async def _trigger_policy(
        self,
        context: _RunContext,
        reason: StopReason,
    ) -> RunResult:
        await self._append(
            context,
            EventType.POLICY_TRIGGERED,
            {
                "policy": reason.value,
                "stop_reason": reason.value,
                "state": context.run.state.value,
                "steps": context.run.steps,
                "elapsed_seconds": self._elapsed(context),
            },
        )
        return await self._terminate(context, RunState.STOPPED, reason)

    async def _terminate(
        self,
        context: _RunContext,
        state: RunState,
        reason: StopReason,
        *,
        output: str | None = None,
        error: str | None = None,
    ) -> RunResult:
        if is_terminal(context.run.state):
            return self._result(context.run)
        validate_transition(context.run.state, state)

        # Preserve a resumable pre-terminal boundary, even when per-step snapshots are off.
        await self._checkpoint(context)
        now = self._now()
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
                "duration_seconds": self._elapsed(context, now=now),
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
        persisted_state, persisted_terminal = await self.event_store.finalize_run(
            terminal_record,
            state_event,
            terminal_event,
        )
        context.run = terminal_record
        await self._event_persisted(persisted_state)
        await self._event_persisted(persisted_terminal)
        return self._result(terminal_record)

    async def _transition(self, context: _RunContext, state: RunState) -> AgentEvent:
        validate_transition(context.run.state, state)
        updated = self._active_record(context, state=state)
        event = AgentEvent(
            run_id=context.run.run_id,
            event_type=EventType.STATE_CHANGED,
            created_at=updated.updated_at,
            payload={
                "from_state": context.run.state.value,
                "to_state": state.value,
            },
        )
        persisted = await self.event_store.append_event_and_update_run(event, updated)
        context.run = updated
        await self._event_persisted(persisted)
        return persisted

    async def _append(
        self,
        context: _RunContext,
        event_type: EventType,
        payload: dict[str, JsonValue],
    ) -> AgentEvent:
        event = AgentEvent(
            run_id=context.run.run_id,
            event_type=event_type,
            created_at=self._now(),
            payload=payload,
        )
        persisted = await self.event_store.append_event(event)
        await self._event_persisted(persisted)
        return persisted

    async def _append_with_run(
        self,
        context: _RunContext,
        event_type: EventType,
        payload: dict[str, JsonValue],
    ) -> AgentEvent:
        event = AgentEvent(
            run_id=context.run.run_id,
            event_type=event_type,
            created_at=self._now(),
            payload=payload,
        )
        persisted = await self.event_store.append_event_and_update_run(event, context.run)
        await self._event_persisted(persisted)
        return persisted

    async def _checkpoint(self, context: _RunContext) -> Checkpoint:
        checkpoint = Checkpoint(
            run_id=context.run.run_id,
            created_at=self._now(),
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
            provider_metadata=self._provider_snapshot(),
            pending_response=context.pending_response,
        )
        event = AgentEvent(
            run_id=context.run.run_id,
            event_type=EventType.CHECKPOINT_CREATED,
            created_at=checkpoint.created_at,
            payload={
                "checkpoint_id": checkpoint.checkpoint_id,
                "state": checkpoint.state.value,
                "next_step": checkpoint.next_step,
            },
        )
        persisted_checkpoint, persisted_event = await self.event_store.save_checkpoint(
            checkpoint,
            event,
        )
        await self._event_persisted(persisted_event)
        return persisted_checkpoint

    async def _reconcile_after_checkpoint(
        self,
        context: _RunContext,
        checkpoint: Checkpoint,
    ) -> None:
        events = await self.event_store.get_events(context.run.run_id)
        trailing = [event for event in events if event.sequence > checkpoint.last_event_sequence]
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
                self._record_tool_result(context, result)
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
            context.run = self._active_record(context)
            await self._transition(context, RunState.OBSERVATION_RECORDED)
            await self._checkpoint(context)

    def _record_tool_result(self, context: _RunContext, result: ToolResult) -> None:
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

    def _provider_snapshot(self) -> dict[str, JsonValue]:
        if isinstance(self.provider, StatefulModelProvider):
            return _JSON_OBJECT_ADAPTER.validate_python(self.provider.snapshot_state())
        return {}

    def _restore_provider(self, metadata: dict[str, JsonValue]) -> None:
        if not metadata:
            return
        if not isinstance(self.provider, StatefulModelProvider):
            raise UnsafeResumeError(
                "The checkpoint contains provider state, but the configured provider "
                "does not implement restore_state()."
            )
        self.provider.restore_state(metadata)

    def _active_record(
        self,
        context: _RunContext,
        *,
        state: RunState | None = None,
        steps: int | None = None,
        error: str | None = None,
    ) -> RunRecord:
        return RunRecord.model_validate(
            {
                **context.run.model_dump(),
                "state": state or context.run.state,
                "steps": context.run.steps if steps is None else steps,
                "token_usage": context.token_usage,
                "token_accounting_available": context.token_accounting_available,
                "user_state": context.user_state,
                "error": error,
                "updated_at": self._now(),
            }
        )

    def _operation_boundary_reason(self, context: _RunContext) -> StopReason | None:
        return context.policy.runtime_reason(self._elapsed(context))

    def _elapsed(self, context: _RunContext, *, now: datetime | None = None) -> float:
        current = now or self._now()
        return max(0.0, (current - context.run.created_at).total_seconds())

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Runtime clock must return a timezone-aware datetime.")
        return value

    @staticmethod
    def _assistant_message(response: ModelResponse) -> dict[str, JsonValue]:
        if response.tool_call is not None:
            return {
                "role": "assistant",
                "tool_call": cast(JsonValue, response.tool_call.model_dump(mode="json")),
            }
        return {"role": "assistant", "content": response.content}

    @staticmethod
    def _require_pending_response(context: _RunContext) -> ModelResponse:
        if context.pending_response is None:
            raise RuntimeError(
                f"State {context.run.state.value!r} requires a pending model response."
            )
        return context.pending_response

    @staticmethod
    def _require_tool_call(response: ModelResponse) -> ToolCall:
        if response.tool_call is None:
            raise RuntimeError("The current model decision does not contain a tool call.")
        return response.tool_call

    @staticmethod
    def _result(run: RunRecord) -> RunResult:
        if run.stop_reason is None:
            raise RuntimeError("Cannot build RunResult before a stop reason is persisted.")
        return RunResult(
            run_id=run.run_id,
            status=run.state,
            stop_reason=run.stop_reason,
            output=run.output,
            error=run.error,
            steps=run.steps,
            token_usage=run.token_usage,
            token_accounting_available=run.token_accounting_available,
        )

    async def _event_persisted(self, event: AgentEvent) -> None:
        """Hook called after each durable event; useful for failure-injection tests."""

        del event
