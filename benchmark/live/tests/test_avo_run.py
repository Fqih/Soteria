"""Deterministic tests for the live benchmark Avo runner and resume harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from avo import ModelRequest, ModelResponse, TokenUsage, ToolCall
from avo.providers import FakeProvider
from avo.state import RunState, StopReason
from benchmark.live.avo_run import (
    avo_contained,
    run_avo,
    run_avo_interrupted,
)
from benchmark.live.scenarios import LiveScenario, scenario_by_name


def _normal_scenario() -> LiveScenario:
    return scenario_by_name("normal_completion")


def _repetition_scenario() -> LiveScenario:
    return scenario_by_name("repetition_prone")


def _interruption_scenario() -> LiveScenario:
    return scenario_by_name("interrupted_resume")


def _expect_no_credentials(payload: dict[str, JsonValue]) -> None:
    serialized = repr(payload).lower()
    for forbidden in ("api_key", "authorization", "secret", "password", "token="):
        assert forbidden not in serialized, f"unexpected credential field present: {forbidden}"


@pytest.mark.asyncio
async def test_run_avo_maps_normal_completion() -> None:
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

    record = await run_avo(provider, _normal_scenario(), run_index=0)

    assert record.scenario == "normal_completion"
    assert record.approach == "avo"
    assert record.run_index == 0
    assert record.status is RunState.COMPLETED
    assert record.stop_reason is StopReason.COMPLETED
    assert record.outcome is None
    assert record.steps == 2
    assert record.token_usage.input_tokens == 9
    assert record.token_usage.output_tokens == 5
    assert record.token_accounting_available is True
    assert record.repeated_action_detected is False
    assert record.manual_step_cap_hit is False
    assert record.resume_tool_executed_exactly_once is None
    assert record.duration_seconds >= 0.0
    assert avo_contained(record) is False

    assert record.trace_text is not None
    assert "Run:" in record.trace_text
    assert "Stop reason: completed" in record.trace_text
    _expect_no_credentials(record.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_run_avo_maps_repetition_policy_stop() -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="call-repeat",
                    name="add",
                    arguments={"left": 7, "right": 5},
                )
            )
        ],
        repeat_last=True,
    )

    record = await run_avo(provider, _repetition_scenario(), run_index=1)

    assert record.scenario == "repetition_prone"
    assert record.approach == "avo"
    assert record.run_index == 1
    assert record.status is RunState.STOPPED
    assert record.stop_reason is StopReason.REPEATED_ACTION
    assert record.repeated_action_detected is True
    assert record.manual_step_cap_hit is False
    assert avo_contained(record) is True
    assert record.trace_text is not None
    assert "Stop reason: repeated_action" in record.trace_text


@pytest.mark.asyncio
async def test_run_avo_marks_token_accounting_unavailable() -> None:
    # FakeProvider substitutes an empty TokenUsage for None, so a bespoke
    # provider is required to exercise the missing-accounting branch.
    class _UsageMissingProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            del request
            return ModelResponse(content="done", usage=None)

    record = await run_avo(_UsageMissingProvider(), _normal_scenario(), run_index=0)

    assert record.status is RunState.COMPLETED
    assert record.token_accounting_available is False


def _resume_provider_factory() -> FakeProvider:
    return FakeProvider(
        [
            ModelResponse(
                tool_call=ToolCall(
                    tool_call_id="side-effect-1",
                    name="record_side_effect",
                    arguments={"marker": "live-resume"},
                ),
                usage=TokenUsage(input_tokens=6, output_tokens=2),
            ),
            ModelResponse(
                content="Recorded live-resume once.",
                usage=TokenUsage(input_tokens=3, output_tokens=1),
            ),
        ]
    )


@pytest.mark.asyncio
async def test_run_avo_interrupted_executes_side_effect_exactly_once() -> None:
    record = await run_avo_interrupted(
        _resume_provider_factory,
        _interruption_scenario(),
        run_index=2,
    )

    assert record.scenario == "interrupted_resume"
    assert record.approach == "avo"
    assert record.run_index == 2
    assert record.status is RunState.COMPLETED
    assert record.stop_reason is StopReason.COMPLETED
    assert record.resume_tool_executed_exactly_once is True
    assert avo_contained(record) is False

    # The side effect must be durably recorded exactly once across the interruption.
    assert record.trace_text is not None
    assert record.trace_text.count("tool_completed") == 1
    assert "resumed from checkpoint" in record.trace_text
    _expect_no_credentials(record.model_dump(mode="json"))
