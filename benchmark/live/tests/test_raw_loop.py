"""Deterministic tests for the live benchmark raw loop and record models."""

from __future__ import annotations

import pytest
from pydantic import JsonValue, ValidationError

from benchmark.live.models import (
    LiveResults,
    LiveRunRecord,
    UnexpectedRawLoopError,
)
from benchmark.live.raw_loop import classify_raw_outcome, run_raw_loop
from benchmark.live.scenarios import LiveScenario, scenario_by_name
from hernness import ModelRequest, ModelResponse, TokenUsage, ToolCall
from hernness.exceptions import ProviderError
from hernness.providers import FakeProvider


def _normal_scenario() -> LiveScenario:
    return scenario_by_name("normal_completion")


def _repetition_scenario() -> LiveScenario:
    return scenario_by_name("repetition_prone")


def _expect_no_credentials(payload: dict[str, JsonValue]) -> None:
    """Reject obvious credential patterns from any serialized record field."""

    serialized = repr(payload).lower()
    for forbidden in ("api_key", "authorization", "secret", "password", "token="):
        assert forbidden not in serialized, f"unexpected credential field present: {forbidden}"


def test_live_run_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LiveRunRecord(
            scenario="normal_completion",
            approach="raw",
            steps=1,
            duration_seconds=0.01,
            token_usage=TokenUsage(),
            token_accounting_available=True,
            leaked_secret="sk-should-not-be-here",  # type: ignore[call-arg]
        )


def test_live_results_round_trip_contains_no_credentials() -> None:
    record = LiveRunRecord(
        scenario="normal_completion",
        approach="raw",
        outcome="completed",
        steps=1,
        duration_seconds=0.01,
        token_usage=TokenUsage(input_tokens=1, output_tokens=2),
        token_accounting_available=True,
    )
    results = LiveResults(
        provider="fake",
        api_style="fake",
        model="fake-model",
        runs=1,
        records=[record],
    )

    payload = results.model_dump(mode="json")
    _expect_no_credentials(payload)

    rebuilt = LiveResults.model_validate(payload)
    assert rebuilt.provider == "fake"
    assert rebuilt.model == "fake-model"
    assert rebuilt.runs == 1
    assert rebuilt.records[0].scenario == "normal_completion"


def test_live_results_rejects_mismatched_runs_count() -> None:
    with pytest.raises(ValidationError):
        LiveResults(
            provider="fake",
            model="fake-model",
            runs=2,
            records=[
                LiveRunRecord(
                    scenario="normal_completion",
                    approach="raw",
                    steps=1,
                    duration_seconds=0.0,
                    token_usage=TokenUsage(),
                    token_accounting_available=True,
                )
            ],
        )


def test_classify_raw_outcome_distinguishes_three_categories() -> None:
    completed = LiveRunRecord(
        scenario="normal_completion",
        approach="raw",
        outcome="completed",
        steps=2,
        duration_seconds=0.01,
        token_usage=TokenUsage(),
        token_accounting_available=True,
    )
    capped = LiveRunRecord(
        scenario="repetition_prone",
        approach="raw",
        outcome="hit_manual_step_cap",
        steps=3,
        duration_seconds=0.01,
        token_usage=TokenUsage(),
        token_accounting_available=True,
        manual_step_cap_hit=True,
    )
    errored = LiveRunRecord(
        scenario="normal_completion",
        approach="raw",
        outcome="error",
        steps=1,
        duration_seconds=0.01,
        token_usage=TokenUsage(),
        token_accounting_available=True,
        expected_error_type="ProviderError",
    )

    assert classify_raw_outcome(completed) == "completed"
    assert classify_raw_outcome(capped) == "hit_manual_step_cap"
    assert classify_raw_outcome(errored) == "error"


@pytest.mark.asyncio
async def test_raw_loop_completes_after_tool_then_final_response() -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="call-add",
                    name="add",
                    arguments={"left": 12, "right": 8},
                ),
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
            ModelResponse(
                content="The sum is 20.",
                usage=TokenUsage(input_tokens=4, output_tokens=2),
            ),
        ]
    )

    record = await run_raw_loop(
        provider, _normal_scenario(), manual_step_cap=5, max_completion_tokens=32
    )

    assert record.scenario == "normal_completion"
    assert record.approach == "raw"
    assert record.outcome == "completed"
    assert record.manual_step_cap_hit is False
    assert record.loop_contained is False
    assert record.steps == 2
    assert record.token_usage.input_tokens == 9
    assert record.token_usage.output_tokens == 5
    assert record.token_accounting_available is True
    assert record.expected_error_type is None
    assert record.unexpected_error_type is None
    assert record.duration_seconds >= 0.0


@pytest.mark.asyncio
async def test_raw_loop_classifier_reports_manual_cap() -> None:
    provider = FakeProvider(
        [ModelResponse(tool_call=ToolCall(name="add", arguments={"left": 1, "right": 1}))],
        repeat_last=True,
    )

    record = await run_raw_loop(
        provider, _repetition_scenario(), manual_step_cap=2, max_completion_tokens=32
    )

    assert record.outcome == "hit_manual_step_cap"
    assert record.manual_step_cap_hit is True
    assert record.loop_contained is False
    assert record.steps == 2
    assert classify_raw_outcome(record) == "hit_manual_step_cap"


@pytest.mark.asyncio
async def test_raw_loop_catches_provider_error_as_expected() -> None:
    provider = FakeProvider([ProviderError("rate limited", retryable=True)])

    record = await run_raw_loop(
        provider, _normal_scenario(), manual_step_cap=5, max_completion_tokens=32
    )

    assert record.outcome == "error"
    assert record.expected_error_type == "ProviderError"
    assert record.unexpected_error_type is None
    assert record.unexpected_error_message is None
    assert record.loop_contained is False
    assert record.steps == 1


@pytest.mark.asyncio
async def test_raw_loop_catches_tool_execution_error_as_expected() -> None:
    from pydantic import BaseModel

    from hernness.tools import FunctionTool

    class _RaisingArgs(BaseModel):
        pass

    async def _impl(arguments: _RaisingArgs) -> dict[str, JsonValue]:
        raise RuntimeError("intentional tool failure")

    raising_tool = FunctionTool(
        name="raising",
        description="A tool that always raises to trigger ToolExecutionError.",
        arguments_model=_RaisingArgs,
        function=_impl,
    )

    def _make_raising_tools(counter: list[int] | None) -> list:
        del counter
        return [raising_tool]

    scenario = _normal_scenario()
    raising_scenario = LiveScenario(
        name=scenario.name,
        task=scenario.task,
        policy=scenario.policy,
        supports_raw=scenario.supports_raw,
        supports_resume=scenario.supports_resume,
        make_tools=_make_raising_tools,
    )

    class _OneCallProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="call-bad",
                    name="raising",
                    arguments={},
                )
            )

    record = await run_raw_loop(
        _OneCallProvider(),
        raising_scenario,
        manual_step_cap=3,
        max_completion_tokens=32,
    )

    assert record.outcome == "error"
    assert record.expected_error_type == "ToolExecutionError"
    assert record.unexpected_error_type is None
    assert record.loop_contained is False
    assert record.steps == 1


@pytest.mark.asyncio
async def test_raw_loop_counts_one_step_per_model_call() -> None:
    provider = FakeProvider(
        [ModelResponse(content="done in one step", usage=TokenUsage(input_tokens=1))]
    )

    record = await run_raw_loop(
        provider, _normal_scenario(), manual_step_cap=5, max_completion_tokens=32
    )

    assert record.steps == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_raw_loop_aggregates_token_usage_across_calls() -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(name="add", arguments={"left": 1, "right": 2}),
                usage=TokenUsage(input_tokens=7, output_tokens=2),
            ),
            ModelResponse(
                content="3",
                usage=TokenUsage(input_tokens=3, output_tokens=1),
            ),
        ]
    )

    record = await run_raw_loop(
        provider, _normal_scenario(), manual_step_cap=5, max_completion_tokens=32
    )

    assert record.outcome == "completed"
    assert record.token_usage.input_tokens == 10
    assert record.token_usage.output_tokens == 3
    assert record.token_accounting_available is True


@pytest.mark.asyncio
async def test_raw_loop_marks_token_accounting_unavailable_when_usage_missing() -> None:
    class _UsageMissingProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(tool_call=ToolCall(name="add", arguments={"left": 1, "right": 1}))

    record = await run_raw_loop(
        _UsageMissingProvider(),
        _normal_scenario(),
        manual_step_cap=2,
        max_completion_tokens=32,
    )

    assert record.token_accounting_available is False
    assert record.outcome == "hit_manual_step_cap"


@pytest.mark.asyncio
async def test_raw_loop_records_loop_contained_false_for_every_outcome() -> None:
    completed = await run_raw_loop(
        FakeProvider([ModelResponse(content="done", usage=TokenUsage())]),
        _normal_scenario(),
        manual_step_cap=5,
        max_completion_tokens=32,
    )
    capped = await run_raw_loop(
        FakeProvider(
            [ModelResponse(tool_call=ToolCall(name="add", arguments={"left": 1, "right": 1}))],
            repeat_last=True,
        ),
        _repetition_scenario(),
        manual_step_cap=2,
        max_completion_tokens=32,
    )
    errored = await run_raw_loop(
        FakeProvider([ProviderError("boom", retryable=False)]),
        _normal_scenario(),
        manual_step_cap=5,
        max_completion_tokens=32,
    )

    for record in (completed, capped, errored):
        assert record.loop_contained is False


@pytest.mark.asyncio
async def test_raw_loop_re_raises_unexpected_exception_with_record() -> None:
    class _BoomProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError("kaboom")

    with pytest.raises(UnexpectedRawLoopError) as exc_info:
        await run_raw_loop(
            _BoomProvider(),
            _normal_scenario(),
            manual_step_cap=5,
            max_completion_tokens=32,
        )

    record = exc_info.value.record
    assert record.outcome == "error"
    assert record.unexpected_error_type == "RuntimeError"
    assert record.unexpected_error_message == "kaboom"
    assert record.expected_error_type is None
    assert record.loop_contained is False


def test_raw_loop_module_does_not_import_forbidden_modules() -> None:
    """The raw loop must avoid AgentRuntime, LoopPolicy, ProgressDetector, and storage."""

    from benchmark.live import raw_loop

    module_source = raw_loop.__file__
    assert module_source is not None
    with open(module_source, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in (
        "hernness.runtime",
        "hernness.policies",
        "hernness.progress",
        "hernness.storage",
    ):
        assert forbidden not in text, f"raw_loop must not import from {forbidden}"
