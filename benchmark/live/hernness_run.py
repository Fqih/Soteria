"""Hernness-managed live benchmark execution, including interruption and resume.

Unlike ``raw_loop``, this runner deliberately drives the full Hernness stack:
``AgentRuntime`` coordinates the provider, the scenario's typed tools, and its
``LoopPolicy`` while an event store records a durable trace. The recorded
``LiveRunRecord`` therefore reflects Hernness's safety net (policy stops and
resume-safe idempotency), not a manual step cap.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.live.models import LiveRunRecord
from benchmark.live.scenarios import LiveScenario
from hernness import AgentEvent, AgentRuntime, EventType, RunResult, RunTrace
from hernness.providers.base import ModelProvider
from hernness.state import RunState, StopReason
from hernness.storage import InMemoryEventStore, SQLiteEventStore

# Policy-driven stop reasons that mean Hernness's containment fences fired.
_CONTAINMENT_STOP_REASONS = frozenset(
    {
        StopReason.MAX_STEPS,
        StopReason.MAX_RUNTIME,
        StopReason.TOKEN_BUDGET_EXCEEDED,
        StopReason.REPEATED_ACTION,
        StopReason.NO_PROGRESS,
        StopReason.CONSECUTIVE_ERRORS,
        StopReason.POLICY_DENIED,
    }
)


class BenchmarkInterruption(BaseException):
    """Simulate abrupt process loss before the runtime can clean up.

    A ``BaseException`` (not ``Exception``) so ``AgentRuntime.run`` does not
    convert it into a ``FAILED`` terminal state; it escapes exactly like a
    killed process would, leaving the persisted trace mid-run.
    """


class InterruptAfterToolCompletedRuntime(AgentRuntime):
    """Raise ``BenchmarkInterruption`` once, right after a tool result is durable."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._interrupted = False

    async def _event_persisted(self, event: AgentEvent) -> None:
        if not self._interrupted and event.event_type is EventType.TOOL_COMPLETED:
            self._interrupted = True
            raise BenchmarkInterruption
        await super()._event_persisted(event)


def hernness_contained(record: LiveRunRecord) -> bool:
    """Return whether Hernness's policy fences contained the run.

    Containment means the runtime reached ``STOPPED`` via a policy stop reason
    (for example a repeated action). It is never inferred from a manual step
    cap, which Hernness does not use.
    """

    return record.status is RunState.STOPPED and record.stop_reason in _CONTAINMENT_STOP_REASONS


def _record_from_result(
    scenario: LiveScenario,
    run_index: int,
    result: RunResult,
    trace: RunTrace,
    *,
    resume_tool_executed_exactly_once: bool | None = None,
) -> LiveRunRecord:
    """Map a terminal ``RunResult`` plus its trace onto a ``LiveRunRecord``."""

    return LiveRunRecord(
        scenario=scenario.name,
        approach="hernness",
        run_index=run_index,
        status=result.status,
        stop_reason=result.stop_reason,
        steps=result.steps,
        duration_seconds=trace.duration_seconds or 0.0,
        token_usage=result.token_usage,
        token_accounting_available=result.token_accounting_available,
        repeated_action_detected=result.stop_reason is StopReason.REPEATED_ACTION,
        manual_step_cap_hit=False,
        resume_tool_executed_exactly_once=resume_tool_executed_exactly_once,
        trace_text=trace.to_text(),
    )


async def run_hernness(
    provider: ModelProvider,
    scenario: LiveScenario,
    run_index: int,
) -> LiveRunRecord:
    """Drive a Hernness-managed run to a terminal state and record its metrics.

    Args:
        provider: The provider consulted for every model decision.
        scenario: The live scenario supplying the task, typed tools, and policy.
        run_index: The zero-based index of this run within a repeated series.

    Returns:
        A ``LiveRunRecord`` mapping the terminal ``RunResult`` and inspected
        trace. Containment is decided by policy stop reasons via
        :func:`hernness_contained`, never by a manual cap.
    """

    event_store = InMemoryEventStore()
    runtime = AgentRuntime(
        provider=provider,
        tools=scenario.tools(),
        policy=scenario.policy,
        event_store=event_store,
    )
    result = await runtime.run(scenario.task)
    trace = await runtime.inspect(result.run_id)
    return _record_from_result(scenario, run_index, result, trace)


async def run_hernness_interrupted(
    provider_factory: Callable[[], ModelProvider],
    scenario: LiveScenario,
    run_index: int,
) -> LiveRunRecord:
    """Interrupt a durable run after a tool result, then resume it from SQLite.

    The first runtime raises :class:`BenchmarkInterruption` immediately after the
    tool result becomes durable. The SQLite store is closed and reopened to model
    a process restart, a fresh provider/runtime is created with the same typed
    tool, and ``resume`` continues from the latest safe checkpoint. The
    scenario's side-effect counter is threaded through both runtimes so the run
    can assert the tool executed exactly once across the interruption.

    Args:
        provider_factory: Builds a fresh provider for each runtime; the resumed
            provider restores its cursor from the persisted checkpoint.
        scenario: The resume-capable scenario supplying the task, tool, policy.
        run_index: The zero-based index of this run within a repeated series.

    Returns:
        A ``LiveRunRecord`` for the resumed terminal run with
        ``resume_tool_executed_exactly_once`` set from the shared counter.
    """

    counter = [0]
    run_id = f"live-hernness-{scenario.name}-{run_index}"

    with TemporaryDirectory(prefix="hernness-live-") as directory:
        database = Path(directory) / "runs.db"

        first_store = SQLiteEventStore(database)
        first_runtime = InterruptAfterToolCompletedRuntime(
            provider=provider_factory(),
            tools=scenario.tools(counter),
            policy=scenario.policy,
            event_store=first_store,
        )
        with contextlib.suppress(BenchmarkInterruption):
            await first_runtime.run(scenario.task, run_id=run_id)
        await first_store.close()

        reopened_store = SQLiteEventStore(database)
        resumed_runtime = AgentRuntime(
            provider=provider_factory(),
            tools=scenario.tools(counter),
            policy=scenario.policy,
            event_store=reopened_store,
        )
        result = await resumed_runtime.resume(run_id)
        trace = await resumed_runtime.inspect(result.run_id)

        events = await reopened_store.get_events(result.run_id)
        tool_completed_count = sum(
            1 for event in events if event.event_type is EventType.TOOL_COMPLETED
        )
        await reopened_store.close()

    if tool_completed_count != 1:
        raise AssertionError(
            f"Expected exactly one durable TOOL_COMPLETED event, found {tool_completed_count}."
        )

    return _record_from_result(
        scenario,
        run_index,
        result,
        trace,
        resume_tool_executed_exactly_once=counter[0] == 1,
    )


__all__ = [
    "BenchmarkInterruption",
    "InterruptAfterToolCompletedRuntime",
    "hernness_contained",
    "run_hernness",
    "run_hernness_interrupted",
]
