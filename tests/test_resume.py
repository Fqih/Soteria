"""Checkpoint restoration and side-effect idempotency tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from avo import (
    AgentEvent,
    AgentRuntime,
    EventType,
    LoopPolicy,
    ModelResponse,
    RunState,
    StopReason,
    TokenUsage,
    ToolCall,
)
from avo.exceptions import (
    CheckpointNotFoundError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    UnsafeResumeError,
)
from avo.providers import FakeProvider
from avo.storage import InMemoryEventStore, SQLiteEventStore
from tests.helpers import seed_run, value_tool


class InjectedInterruption(BaseException):
    """Simulate abrupt process loss without runtime cleanup."""


class InterruptOnEventRuntime(AgentRuntime):
    """Interrupt once after a selected event has become durable."""

    def __init__(self, *args: object, interrupt_event: EventType, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._interrupt_event = interrupt_event
        self._interrupted = False

    async def _event_persisted(self, event: AgentEvent) -> None:
        if not self._interrupted and event.event_type is self._interrupt_event:
            self._interrupted = True
            raise InjectedInterruption


class InterruptAfterToolCheckpointRuntime(AgentRuntime):
    """Interrupt after the observation checkpoint is fully persisted."""

    async def _event_persisted(self, event: AgentEvent) -> None:
        if (
            event.event_type is EventType.CHECKPOINT_CREATED
            and event.payload.get("state") == RunState.OBSERVATION_RECORDED.value
        ):
            raise InjectedInterruption


@pytest.mark.asyncio
async def test_resume_unknown_run_is_rejected() -> None:
    runtime = AgentRuntime(provider=FakeProvider([]))

    with pytest.raises(RunNotFoundError):
        await runtime.resume("unknown")


@pytest.mark.asyncio
async def test_resume_terminal_run_is_rejected() -> None:
    runtime = AgentRuntime(provider=FakeProvider([ModelResponse(content="done")]))
    result = await runtime.run("finish")

    with pytest.raises(RunAlreadyTerminalError, match="terminal"):
        await runtime.resume(result.run_id)


@pytest.mark.asyncio
async def test_resume_without_checkpoint_has_actionable_error() -> None:
    store = InMemoryEventStore()
    await seed_run(store, run_id="no-checkpoint")
    runtime = AgentRuntime(provider=FakeProvider([]), event_store=store)

    with pytest.raises(CheckpointNotFoundError, match="successfully persisted"):
        await runtime.resume("no-checkpoint")


@pytest.mark.asyncio
async def test_interrupt_after_tool_result_resumes_without_duplicate_side_effect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resume.db"
    first_store = SQLiteEventStore(path)
    counter = [0]
    tool = value_tool(counter=counter)
    first_provider = FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="side-effect-1",
                    name="value",
                    arguments={"value": 9},
                ),
                usage=TokenUsage(input_tokens=3, output_tokens=1),
            ),
            ModelResponse(
                content="resumed safely",
                usage=TokenUsage(input_tokens=2, output_tokens=2),
            ),
        ]
    )
    first_runtime = InterruptOnEventRuntime(
        provider=first_provider,
        tools=[tool],
        event_store=first_store,
        interrupt_event=EventType.TOOL_COMPLETED,
    )

    with pytest.raises(InjectedInterruption):
        await first_runtime.run("side effect once", run_id="resumable-run")
    assert counter[0] == 1
    await first_store.close()

    second_store = SQLiteEventStore(path)
    second_runtime = AgentRuntime(
        provider=FakeProvider([]),
        tools=[tool],
        event_store=second_store,
    )
    result = await second_runtime.resume("resumable-run")
    events = await second_store.get_events("resumable-run")

    assert result.status is RunState.COMPLETED
    assert result.output == "resumed safely"
    assert result.token_usage == TokenUsage(input_tokens=5, output_tokens=3)
    assert counter[0] == 1
    assert sum(event.event_type is EventType.TOOL_COMPLETED for event in events) == 1
    assert sum(event.event_type is EventType.RUN_RESUMED for event in events) == 1
    requested = next(event for event in events if event.event_type is EventType.TOOL_REQUESTED)
    started = next(event for event in events if event.event_type is EventType.TOOL_STARTED)
    assert requested.payload["idempotency_key"] == started.payload["idempotency_key"]
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    await second_store.close()


@pytest.mark.asyncio
async def test_interrupt_after_successful_checkpoint_resumes_without_duplicate() -> None:
    store = InMemoryEventStore()
    counter = [0]
    tool = value_tool(counter=counter)
    runtime = InterruptAfterToolCheckpointRuntime(
        provider=FakeProvider(
            [
                ModelResponse(
                    tool_call=ToolCall(
                        tool_call_id="checkpointed-call",
                        name="value",
                        arguments={"value": 1},
                    )
                ),
                ModelResponse(content="done"),
            ]
        ),
        tools=[tool],
        event_store=store,
    )

    with pytest.raises(InjectedInterruption):
        await runtime.run("checkpoint interruption", run_id="checkpoint-run")
    resumed = AgentRuntime(
        provider=FakeProvider([]),
        tools=[tool],
        event_store=store,
    )
    result = await resumed.resume("checkpoint-run")

    assert result.stop_reason is StopReason.COMPLETED
    assert counter[0] == 1


@pytest.mark.asyncio
async def test_started_tool_without_result_is_not_retried_automatically() -> None:
    store = InMemoryEventStore()
    counter = [0]
    tool = value_tool(counter=counter)
    runtime = InterruptOnEventRuntime(
        provider=FakeProvider(
            [
                ModelResponse(
                    tool_call=ToolCall(
                        tool_call_id="uncertain-call",
                        name="value",
                        arguments={"value": 1},
                    )
                )
            ]
        ),
        tools=[tool],
        event_store=store,
        interrupt_event=EventType.TOOL_STARTED,
    )

    with pytest.raises(InjectedInterruption):
        await runtime.run("uncertain side effect", run_id="uncertain-run")
    assert counter[0] == 0

    resumed = AgentRuntime(
        provider=FakeProvider([]),
        tools=[tool],
        event_store=store,
    )
    with pytest.raises(UnsafeResumeError, match="started without a durable result"):
        await resumed.resume("uncertain-run")
    assert counter[0] == 0


@pytest.mark.asyncio
async def test_checkpoint_every_step_false_still_checkpoints_tool_result() -> None:
    store = InMemoryEventStore()
    runtime = AgentRuntime(
        provider=FakeProvider(
            [
                ModelResponse(
                    tool_call=ToolCall(
                        tool_call_id="mandatory-checkpoint",
                        name="value",
                        arguments={"value": 1},
                    )
                ),
                ModelResponse(content="done"),
            ]
        ),
        tools=[value_tool()],
        policy=LoopPolicy(checkpoint_every_step=False),
        event_store=store,
    )

    result = await runtime.run("minimum checkpoint guarantee")
    events = await store.get_events(result.run_id)
    tool_result_sequence = next(
        event.sequence for event in events if event.event_type is EventType.TOOL_COMPLETED
    )
    later_checkpoints = [
        event
        for event in events
        if event.event_type is EventType.CHECKPOINT_CREATED
        and event.sequence > tool_result_sequence
    ]

    assert later_checkpoints
    assert any(
        event.payload.get("state") == RunState.OBSERVATION_RECORDED.value
        for event in later_checkpoints
    )


@pytest.mark.asyncio
async def test_resume_uses_persisted_policy_snapshot() -> None:
    store = InMemoryEventStore()
    counter = [0]
    tool = value_tool(counter=counter)
    runtime = InterruptOnEventRuntime(
        provider=FakeProvider(
            [
                ModelResponse(
                    tool_call=ToolCall(
                        tool_call_id="one-step-call",
                        name="value",
                        arguments={"value": 1},
                    )
                ),
                ModelResponse(content="must not run"),
            ]
        ),
        tools=[tool],
        policy=LoopPolicy(max_steps=1),
        event_store=store,
        interrupt_event=EventType.TOOL_COMPLETED,
    )
    with pytest.raises(InjectedInterruption):
        await runtime.run("persist policy", run_id="policy-run")

    resumed = AgentRuntime(
        provider=FakeProvider([]),
        tools=[tool],
        policy=LoopPolicy(max_steps=99),
        event_store=store,
    )
    result = await resumed.resume("policy-run")

    assert result.stop_reason is StopReason.MAX_STEPS
    assert counter[0] == 1
