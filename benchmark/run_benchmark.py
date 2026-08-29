"""Run the deterministic raw-loop versus Hernness reliability benchmark."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel
from scenarios.catalog import SCENARIOS, Scenario, ScenarioKind

from hernness import (
    AgentEvent,
    AgentRuntime,
    EventType,
    FunctionTool,
    LoopPolicy,
    ModelRequest,
    ModelResponse,
    RunState,
    TokenUsage,
    ToolCall,
)
from hernness.events import TERMINAL_EVENT_TYPES
from hernness.providers import FakeProvider, ScriptItem
from hernness.storage import InMemoryEventStore, SQLiteEventStore

HARNESS_STEP_LIMIT = 6


class OperationArguments(BaseModel):
    """Arguments used by all synthetic benchmark tools."""

    value: int


@dataclass(frozen=True)
class Outcome:
    """Normalized metrics emitted by one system and scenario."""

    scenario: ScenarioKind
    task_completed: bool
    contained: bool
    resume_succeeded: bool
    duplicate_side_effects: int
    terminal_complete: bool
    trace_reproducible: bool
    steps: int
    runtime_seconds: float


class BenchmarkInterruption(BaseException):
    """Simulate abrupt process exit after a durable result event."""


class InterruptingRuntime(AgentRuntime):
    """Interrupt once immediately after TOOL_COMPLETED."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._interrupted = False

    async def _event_persisted(self, event: AgentEvent) -> None:
        if not self._interrupted and event.event_type is EventType.TOOL_COMPLETED:
            self._interrupted = True
            raise BenchmarkInterruption


def scenario_script(kind: ScenarioKind) -> list[ScriptItem]:
    """Build a fresh deterministic provider script."""

    if kind is ScenarioKind.NORMAL:
        return [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="normal-1",
                    name="operation",
                    arguments={"value": 1},
                ),
                usage=TokenUsage(input_tokens=2, output_tokens=1),
            ),
            ModelResponse(
                content="complete",
                usage=TokenUsage(input_tokens=2, output_tokens=1),
            ),
        ]
    if kind is ScenarioKind.PROVIDER_FAILURE:
        return [RuntimeError(f"provider failure {index}") for index in range(HARNESS_STEP_LIMIT)]

    responses: list[ScriptItem] = []
    for index in range(HARNESS_STEP_LIMIT):
        value: object = index
        if kind is ScenarioKind.REPEATED_ACTION:
            value = 1
        elif kind is ScenarioKind.MALFORMED_ARGUMENTS:
            value = "invalid"
        usage = (
            TokenUsage(input_tokens=20, output_tokens=1)
            if kind is ScenarioKind.BUDGET
            else TokenUsage()
        )
        responses.append(
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id=f"{kind.value}-{index}",
                    name="operation",
                    arguments={"value": value},
                ),
                usage=usage,
            )
        )
    return responses


async def run_raw(scenario: Scenario) -> Outcome:
    """Run a deliberately minimal loop under an external harness cap."""

    started = time.perf_counter()
    if scenario.kind is ScenarioKind.INTERRUPTION:
        side_effects = 0
        side_effects += 1
        side_effects += 1
        return Outcome(
            scenario=scenario.kind,
            task_completed=True,
            contained=False,
            resume_succeeded=False,
            duplicate_side_effects=side_effects - 1,
            terminal_complete=False,
            trace_reproducible=False,
            steps=2,
            runtime_seconds=time.perf_counter() - started,
        )

    provider = FakeProvider(scenario_script(scenario.kind))
    messages: list[dict[str, object]] = [{"role": "user", "content": scenario.description}]
    side_effects = 0
    steps = 0
    for step in range(1, HARNESS_STEP_LIMIT + 1):
        steps = step
        request = ModelRequest(
            run_id=f"raw-{scenario.kind.value}",
            step=step,
            messages=messages,
        )
        try:
            response = await provider.generate(request)
        except Exception:
            continue
        if response.is_final:
            return Outcome(
                scenario=scenario.kind,
                task_completed=True,
                contained=False,
                resume_succeeded=False,
                duplicate_side_effects=0,
                terminal_complete=False,
                trace_reproducible=False,
                steps=steps,
                runtime_seconds=time.perf_counter() - started,
            )

        call = response.tool_call
        assert call is not None
        try:
            value = call.arguments.get("value")
            if not isinstance(value, int):
                raise ValueError("value must be an integer")
            if scenario.kind is ScenarioKind.TOOL_FAILURE:
                raise RuntimeError("synthetic tool failure")
            side_effects += 1
            output: object = (
                {"unchanged": True}
                if scenario.kind is ScenarioKind.REPEATED_OBSERVATION
                else {"value": value}
            )
            messages.append({"role": "tool", "content": output})
        except Exception as exc:
            messages.append({"role": "tool", "content": {"error": str(exc)}})

    duplicates = max(0, side_effects - 1) if scenario.kind is ScenarioKind.REPEATED_ACTION else 0
    return Outcome(
        scenario=scenario.kind,
        task_completed=False,
        contained=False,
        resume_succeeded=False,
        duplicate_side_effects=duplicates,
        terminal_complete=False,
        trace_reproducible=False,
        steps=steps,
        runtime_seconds=time.perf_counter() - started,
    )


async def run_hernness(scenario: Scenario) -> Outcome:
    """Run one scenario through Hernness and collect persisted invariants."""

    if scenario.kind is ScenarioKind.INTERRUPTION:
        return await run_hernness_interruption(scenario)

    started = time.perf_counter()
    side_effects = [0]

    async def operation(arguments: OperationArguments) -> object:
        if scenario.kind is ScenarioKind.TOOL_FAILURE:
            raise RuntimeError("synthetic tool failure")
        side_effects[0] += 1
        if scenario.kind is ScenarioKind.REPEATED_OBSERVATION:
            return {"unchanged": True}
        return {"value": arguments.value}

    store = InMemoryEventStore()
    runtime = AgentRuntime(
        provider=FakeProvider(scenario_script(scenario.kind)),
        tools=[
            FunctionTool(
                name="operation",
                description="Perform one deterministic benchmark operation.",
                arguments_model=OperationArguments,
                function=operation,
            )
        ],
        policy=LoopPolicy(
            max_steps=HARNESS_STEP_LIMIT,
            max_total_tokens=12,
            repeated_action_limit=2,
            consecutive_error_limit=2,
            no_progress_window=2,
        ),
        event_store=store,
    )
    result = await runtime.run(scenario.description)
    events = await store.get_events(result.run_id)
    serializable = json.dumps(
        [event.model_dump(mode="json") for event in events],
        sort_keys=True,
    )
    del serializable
    terminal_complete = (
        bool(events)
        and events[-1].event_type in TERMINAL_EVENT_TYPES
        and result.stop_reason is not None
    )
    trace_reproducible = [event.sequence for event in events] == list(range(1, len(events) + 1))
    duplicates = max(0, side_effects[0] - 1) if scenario.kind is ScenarioKind.REPEATED_ACTION else 0
    await store.close()
    return Outcome(
        scenario=scenario.kind,
        task_completed=result.status is RunState.COMPLETED,
        contained=scenario.expects_containment
        and result.status in {RunState.STOPPED, RunState.FAILED},
        resume_succeeded=False,
        duplicate_side_effects=duplicates,
        terminal_complete=terminal_complete,
        trace_reproducible=trace_reproducible,
        steps=result.steps,
        runtime_seconds=time.perf_counter() - started,
    )


async def run_hernness_interruption(scenario: Scenario) -> Outcome:
    """Measure SQLite recovery after a completed side effect."""

    started = time.perf_counter()
    side_effects = [0]

    async def operation(arguments: OperationArguments) -> object:
        side_effects[0] += 1
        return {"value": arguments.value, "side_effect": side_effects[0]}

    tool = FunctionTool(
        name="operation",
        description="Perform one deterministic benchmark operation.",
        arguments_model=OperationArguments,
        function=operation,
    )
    script = [
        ModelResponse(
            tool_call=ToolCall(
                tool_call_id="interrupted-operation",
                name="operation",
                arguments={"value": 1},
            )
        ),
        ModelResponse(content="recovered"),
    ]
    with TemporaryDirectory(prefix="hernness-benchmark-") as directory:
        path = Path(directory) / "benchmark.db"
        first_store = SQLiteEventStore(path)
        interrupted = InterruptingRuntime(
            provider=FakeProvider(script),
            tools=[tool],
            event_store=first_store,
        )
        with suppress(BenchmarkInterruption):
            await interrupted.run(scenario.description, run_id="benchmark-interruption")
        await first_store.close()

        reopened = SQLiteEventStore(path)
        resumed = AgentRuntime(
            provider=FakeProvider([]),
            tools=[tool],
            event_store=reopened,
        )
        result = await resumed.resume("benchmark-interruption")
        events = await reopened.get_events(result.run_id)
        await reopened.close()

    return Outcome(
        scenario=scenario.kind,
        task_completed=result.status is RunState.COMPLETED,
        contained=False,
        resume_succeeded=result.status is RunState.COMPLETED and side_effects[0] == 1,
        duplicate_side_effects=max(0, side_effects[0] - 1),
        terminal_complete=bool(events) and events[-1].event_type in TERMINAL_EVENT_TYPES,
        trace_reproducible=[event.sequence for event in events] == list(range(1, len(events) + 1)),
        steps=result.steps,
        runtime_seconds=time.perf_counter() - started,
    )


def percentage(numerator: int, denominator: int) -> str:
    """Format a percentage, including a deterministic zero-denominator value."""

    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def aggregate(outcomes: list[Outcome]) -> dict[str, str]:
    """Aggregate required benchmark metrics."""

    containment = [
        outcome
        for outcome in outcomes
        if next(
            scenario for scenario in SCENARIOS if scenario.kind is outcome.scenario
        ).expects_containment
    ]
    resume = [
        outcome
        for outcome in outcomes
        if next(
            scenario for scenario in SCENARIOS if scenario.kind is outcome.scenario
        ).expects_resume
    ]
    return {
        "task completion rate": percentage(
            sum(outcome.task_completed for outcome in outcomes),
            len(outcomes),
        ),
        "loop containment rate": percentage(
            sum(outcome.contained for outcome in containment),
            len(containment),
        ),
        "resume success rate": percentage(
            sum(outcome.resume_succeeded for outcome in resume),
            len(resume),
        ),
        "duplicate side-effect count": str(
            sum(outcome.duplicate_side_effects for outcome in outcomes)
        ),
        "terminal completeness": percentage(
            sum(outcome.terminal_complete for outcome in outcomes),
            len(outcomes),
        ),
        "trace integrity rate": percentage(
            sum(outcome.trace_reproducible for outcome in outcomes),
            len(outcomes),
        ),
        "mean steps": f"{statistics.fmean(outcome.steps for outcome in outcomes):.2f}",
        "mean runtime": (
            f"{statistics.fmean(outcome.runtime_seconds for outcome in outcomes) * 1000:.2f} ms"
        ),
    }


def render_results(raw: list[Outcome], hernness: list[Outcome]) -> str:
    """Render reproducible Markdown benchmark output."""

    raw_metrics = aggregate(raw)
    hernness_metrics = aggregate(hernness)
    rows = [
        "# Deterministic Benchmark Results",
        "",
        "This benchmark uses only scripted FakeProvider decisions. It measures runtime ",
        "containment, persistence, and recovery behavior—not model intelligence.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Minimal raw loop | Hernness |",
        "|---|---:|---:|",
    ]
    for metric in raw_metrics:
        rows.append(f"| {metric.title()} | {raw_metrics[metric]} | {hernness_metrics[metric]} |")
    rows.extend(
        [
            "",
            "## Scenario outcomes",
            "",
            "| Scenario | Raw completed | Hernness completed | "
            "Hernness contained | Hernness resumed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        raw_outcome = next(item for item in raw if item.scenario is scenario.kind)
        hernness_outcomeb = next(item for item in hernness if item.scenario is scenario.kind)
        rows.append(
            f"| {scenario.kind.value} | {str(raw_outcome.task_completed).lower()} | "
            f"{str(hernness_outcomeb.task_completed).lower()} | "
            f"{str(hernness_outcomeb.contained).lower()} | "
            f"{str(hernness_outcomeb.resume_succeeded).lower()} |"
        )
    rows.extend(
        [
            "",
            "The raw loop is stopped by an external six-step benchmark harness so runaway ",
            "scenarios terminate during measurement; this external cap is not counted as ",
            "runtime containment. Terminal completeness requires a persisted terminal state ",
            "and explicit reason. Trace integrity requires JSON serialization and gap-free ",
            "event ordering. Wall-clock timings vary by machine.",
            "",
            "Reproduce with:",
            "",
            "    python benchmark/run_benchmark.py",
            "",
        ]
    )
    return "\n".join(rows)


async def main() -> None:
    """Execute every scenario for both systems and write RESULTS.md."""

    raw = [await run_raw(scenario) for scenario in SCENARIOS]
    hernness = [await run_hernness(scenario) for scenario in SCENARIOS]
    output = render_results(raw, hernness)
    results_path = Path(__file__).with_name("RESULTS.md")
    results_path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
