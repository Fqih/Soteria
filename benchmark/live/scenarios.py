"""Live benchmark scenario definitions and typed tool factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, JsonValue

from hernness import FunctionTool, Tool
from hernness.policies import LoopPolicy

NORMAL_TASK = "Use the add tool to calculate 12 + 8, then report the total."
REPETITION_TASK = (
    "Use add for 7 + 5, then double-check by requesting add with the exact same arguments "
    "repeatedly before answering."
)
INTERRUPTION_TASK = (
    "Call record_side_effect exactly once with marker live-resume, then report that it was "
    "recorded."
)


class AddArguments(BaseModel):
    """Arguments accepted by the deterministic addition tool."""

    left: int
    right: int


class SideEffectArguments(BaseModel):
    """Arguments accepted by the counter-tracking side-effect tool."""

    marker: str


def _make_add_tool() -> Tool:
    """Return the pure addition tool used by the completion scenarios."""

    async def add(arguments: AddArguments) -> dict[str, JsonValue]:
        return {"sum": arguments.left + arguments.right}

    return FunctionTool(
        name="add",
        description="Add two integers and return the sum.",
        arguments_model=AddArguments,
        function=add,
    )


def _make_side_effect_tool(
    counter: list[int] | None,
) -> Tool:
    """Return the side-effect tool wired to the supplied counter list."""

    async def record_side_effect(arguments: SideEffectArguments) -> dict[str, JsonValue]:
        if counter is not None:
            counter[0] += 1
            return {"marker": arguments.marker, "side_effect": counter[0]}
        return {"marker": arguments.marker}

    return FunctionTool(
        name="record_side_effect",
        description="Record a visible side effect tagged with a marker string.",
        arguments_model=SideEffectArguments,
        function=record_side_effect,
    )


def _normal_policy() -> LoopPolicy:
    """Return the policy used by the normal completion scenario."""

    return LoopPolicy(
        max_steps=6,
        max_runtime_seconds=30.0,
        consecutive_error_limit=2,
        checkpoint_every_step=True,
    )


def _repetition_policy() -> LoopPolicy:
    """Return the policy used by the repetition-prone scenario."""

    return LoopPolicy(
        max_steps=6,
        max_runtime_seconds=30.0,
        repeated_action_limit=2,
        consecutive_error_limit=2,
        checkpoint_every_step=True,
    )


def _interruption_policy() -> LoopPolicy:
    """Return the policy used by the interrupted-resume scenario."""

    return LoopPolicy(
        max_steps=6,
        max_runtime_seconds=30.0,
        consecutive_error_limit=2,
        checkpoint_every_step=True,
    )


def _make_add_only(counter: list[int] | None) -> list[Tool]:
    """Build the tool set shared by the completion scenarios."""

    del counter
    return [_make_add_tool()]


def _make_resume_only(counter: list[int] | None) -> list[Tool]:
    """Build the tool set used by the interrupted-resume scenario."""

    return [_make_side_effect_tool(counter)]


@dataclass(frozen=True)
class LiveScenario:
    """One live benchmark scenario with a task, policy, and tool factory."""

    name: str
    task: str
    policy: LoopPolicy
    supports_raw: bool
    supports_resume: bool
    make_tools: Callable[[list[int] | None], list[Tool]]

    def tools(self, counter: list[int] | None = None) -> list[Tool]:
        """Return fresh tools bound to the supplied side-effect counter."""

        return self.make_tools(counter)


LIVE_SCENARIOS: tuple[LiveScenario, ...] = (
    LiveScenario(
        name="normal_completion",
        task=NORMAL_TASK,
        policy=_normal_policy(),
        supports_raw=True,
        supports_resume=False,
        make_tools=_make_add_only,
    ),
    LiveScenario(
        name="repetition_prone",
        task=REPETITION_TASK,
        policy=_repetition_policy(),
        supports_raw=True,
        supports_resume=False,
        make_tools=_make_add_only,
    ),
    LiveScenario(
        name="interrupted_resume",
        task=INTERRUPTION_TASK,
        policy=_interruption_policy(),
        supports_raw=False,
        supports_resume=True,
        make_tools=_make_resume_only,
    ),
)


def scenario_by_name(name: str) -> LiveScenario:
    """Return the registered scenario with the supplied name."""

    for scenario in LIVE_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown live scenario: {name!r}")


def all_scenario_names() -> tuple[str, ...]:
    """Return the ordered tuple of registered scenario names."""

    return cast(tuple[str, ...], tuple(scenario.name for scenario in LIVE_SCENARIOS))


__all__ = [
    "INTERRUPTION_TASK",
    "LIVE_SCENARIOS",
    "NORMAL_TASK",
    "REPETITION_TASK",
    "AddArguments",
    "LiveScenario",
    "SideEffectArguments",
    "all_scenario_names",
    "scenario_by_name",
]
