"""Deterministic tests for live benchmark scenario definitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.live.scenarios import (
    LIVE_SCENARIOS,
    AddArguments,
    LiveScenario,
    SideEffectArguments,
    scenario_by_name,
)
from hernness import FunctionTool, Tool
from hernness.exceptions import ToolValidationError
from hernness.policies import LoopPolicy


def _find(name: str) -> LiveScenario:
    return scenario_by_name(name)


def test_live_scenarios_define_exactly_three_scenarios() -> None:
    assert len(LIVE_SCENARIOS) == 3


def test_live_scenarios_have_unique_names() -> None:
    names = [scenario.name for scenario in LIVE_SCENARIOS]
    assert len(names) == len(set(names))


def test_live_scenarios_use_finite_policy_limits() -> None:
    for scenario in LIVE_SCENARIOS:
        assert scenario.policy.max_steps > 0
        assert scenario.policy.max_runtime_seconds is not None
        assert scenario.policy.max_runtime_seconds > 0
        assert scenario.policy.max_steps <= 100


def test_repetition_scenario_uses_repeated_action_limit_two() -> None:
    scenario = _find("repetition_prone")

    assert scenario.policy.repeated_action_limit == 2


def test_interruption_scenario_is_resume_only() -> None:
    scenario = _find("interrupted_resume")

    assert scenario.supports_raw is False
    assert scenario.supports_resume is True


def test_normal_scenario_supports_raw_loop_only() -> None:
    scenario = _find("normal_completion")

    assert scenario.supports_raw is True
    assert scenario.supports_resume is False


def test_every_scenario_has_non_empty_task_and_policy() -> None:
    for scenario in LIVE_SCENARIOS:
        assert scenario.task.strip()
        assert isinstance(scenario.policy, LoopPolicy)


def test_scenario_by_name_returns_matching_scenario() -> None:
    for scenario in LIVE_SCENARIOS:
        assert scenario_by_name(scenario.name) is scenario


def test_scenario_by_name_raises_for_unknown_name() -> None:
    with pytest.raises(KeyError):
        scenario_by_name("does_not_exist")


@pytest.mark.asyncio
async def test_add_tool_uses_pydantic_arguments_and_returns_sum() -> None:
    scenario = _find("normal_completion")
    tools = scenario.make_tools(None)
    add = next(tool for tool in tools if tool.metadata.name == "add")

    assert isinstance(add, FunctionTool)

    schema = add.metadata.input_schema
    required = schema.get("required")
    assert isinstance(required, list)
    assert "left" in required
    assert "right" in required

    result = await add.invoke({"left": 2, "right": 3})
    assert result == {"sum": 5}


@pytest.mark.asyncio
async def test_add_tool_rejects_malformed_arguments() -> None:
    scenario = _find("normal_completion")
    add = next(tool for tool in scenario.make_tools(None) if tool.metadata.name == "add")

    with pytest.raises(ToolValidationError):
        await add.invoke({"left": "two", "right": 3})


def test_add_arguments_model_rejects_non_integer_left() -> None:
    with pytest.raises(ValidationError):
        AddArguments.model_validate({"left": "two", "right": 3})


@pytest.mark.asyncio
async def test_add_tool_does_not_increment_side_effect_counter() -> None:
    scenario = _find("normal_completion")
    counter = [0]
    add = next(tool for tool in scenario.make_tools(counter) if tool.metadata.name == "add")

    await add.invoke({"left": 4, "right": 6})

    assert counter[0] == 0


@pytest.mark.asyncio
async def test_add_tool_is_shared_across_scenarios_with_same_signature() -> None:
    scenarios = [_find("normal_completion"), _find("repetition_prone")]
    for scenario in scenarios:
        tools = scenario.make_tools(None)
        add = next(tool for tool in tools if tool.metadata.name == "add")
        result = await add.invoke({"left": 7, "right": 5})
        assert result == {"sum": 12}


@pytest.mark.asyncio
async def test_record_side_effect_tool_increments_counter() -> None:
    scenario = _find("interrupted_resume")
    counter = [0]
    record = next(
        tool for tool in scenario.make_tools(counter) if tool.metadata.name == "record_side_effect"
    )

    result = await record.invoke({"marker": "live-resume"})

    assert counter[0] == 1
    assert result == {"marker": "live-resume", "side_effect": 1}


@pytest.mark.asyncio
async def test_record_side_effect_tool_skips_counter_when_unavailable() -> None:
    scenario = _find("interrupted_resume")
    record = next(
        tool for tool in scenario.make_tools(None) if tool.metadata.name == "record_side_effect"
    )

    result = await record.invoke({"marker": "live-resume"})

    assert result == {"marker": "live-resume"}


@pytest.mark.asyncio
async def test_record_side_effect_tool_validates_arguments() -> None:
    scenario = _find("interrupted_resume")
    record = next(
        tool for tool in scenario.make_tools([0]) if tool.metadata.name == "record_side_effect"
    )

    with pytest.raises(ToolValidationError):
        await record.invoke({})


def test_side_effect_arguments_model_rejects_missing_marker() -> None:
    with pytest.raises(ValidationError):
        SideEffectArguments.model_validate({})


def test_make_tools_returns_a_list_of_tools() -> None:
    for scenario in LIVE_SCENARIOS:
        tools = scenario.make_tools(None)
        assert isinstance(tools, list)
        assert tools
        for tool in tools:
            assert isinstance(tool, Tool)
