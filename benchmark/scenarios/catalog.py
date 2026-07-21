"""Scenario metadata shared by the deterministic benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScenarioKind(StrEnum):
    """Kinds of runtime behavior exercised by the benchmark."""

    NORMAL = "normal_completion"
    REPEATED_ACTION = "repeated_tool_calls"
    REPEATED_OBSERVATION = "repeated_observations"
    PROVIDER_FAILURE = "provider_failures"
    TOOL_FAILURE = "tool_failures"
    INTERRUPTION = "interruption_after_side_effect"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    BUDGET = "budget_exhaustion"


@dataclass(frozen=True)
class Scenario:
    """One deterministic benchmark scenario."""

    kind: ScenarioKind
    description: str
    expects_containment: bool = False
    expects_resume: bool = False


SCENARIOS = (
    Scenario(ScenarioKind.NORMAL, "One tool call followed by normal completion."),
    Scenario(
        ScenarioKind.REPEATED_ACTION,
        "Identical tool actions continue without useful progress.",
        expects_containment=True,
    ),
    Scenario(
        ScenarioKind.REPEATED_OBSERVATION,
        "Different actions keep returning an identical observation.",
        expects_containment=True,
    ),
    Scenario(
        ScenarioKind.PROVIDER_FAILURE,
        "Every provider request fails.",
        expects_containment=True,
    ),
    Scenario(
        ScenarioKind.TOOL_FAILURE,
        "Every valid tool invocation fails.",
        expects_containment=True,
    ),
    Scenario(
        ScenarioKind.INTERRUPTION,
        "The process exits immediately after a side effect.",
        expects_resume=True,
    ),
    Scenario(
        ScenarioKind.MALFORMED_ARGUMENTS,
        "Every tool call contains malformed arguments.",
        expects_containment=True,
    ),
    Scenario(
        ScenarioKind.BUDGET,
        "Reported usage exceeds the configured token budget.",
        expects_containment=True,
    ),
)
