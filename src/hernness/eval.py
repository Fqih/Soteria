"""Golden-prompt eval framework.

Define test cases (input prompt + expected output assertions) and run
them against a provider implementation. Designed for offline CI — no
network calls, no API keys. Assertions are pure substring / equality
checks over the captured provider response and tool-call trace.

The reporter returns pass/fail per case plus a summary. Failures are
never silent — the suite aborts at the first failed case unless
``continue_on_failure=True``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from hernness.exceptions import SoteriaError

EvalError = SoteriaError


@dataclass(frozen=True)
class ToolCallExpectation:
    """A tool-call we expect the model to make."""

    name: str
    arguments_subset: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    """One golden-prompt test case."""

    name: str
    prompt: str
    expected_text_substrings: tuple[str, ...] = ()
    expected_tool_calls: tuple[ToolCallExpectation, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalResult:
    """Outcome of running one case."""

    case: EvalCase
    passed: bool
    failures: tuple[str, ...] = ()
    response_text: str = ""


@dataclass(frozen=True)
class EvalReport:
    """Aggregate across a suite."""

    results: tuple[EvalResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def run_case(
    case: EvalCase,
    *,
    run_provider: Any,
) -> EvalResult:
    """Run a single case against a provider-returning callable."""

    failures: list[str] = []
    try:
        response = run_provider(case.prompt)
    except Exception as exc:  # provider errors are test failures, not crashes
        return EvalResult(
            case=case,
            passed=False,
            failures=(f"provider raised {type(exc).__name__}: {exc}",),
        )

    text = str(response.get("content", "")) if isinstance(response, dict) else ""
    tool_calls = response.get("tool_calls", ()) if isinstance(response, dict) else ()

    for needle in case.expected_text_substrings:
        if needle not in text:
            failures.append(f"missing text substring: {needle!r}")

    actual_names = tuple(tc.get("name", "") for tc in tool_calls if isinstance(tc, dict))
    for expected in case.expected_tool_calls:
        if expected.name not in actual_names:
            failures.append(f"missing tool call: {expected.name!r}")
            continue
        matching = next(
            (tc for tc in tool_calls if isinstance(tc, dict) and tc.get("name") == expected.name),
            None,
        )
        if matching is None:
            continue
        for key, want in expected.arguments_subset.items():
            got = (
                matching.get("arguments", {}).get(key)
                if isinstance(matching.get("arguments"), dict)
                else None
            )
            if got != want:
                failures.append(
                    f"tool {expected.name!r} argument {key!r}: expected {want!r}, got {got!r}"
                )

    return EvalResult(
        case=case,
        passed=not failures,
        failures=tuple(failures),
        response_text=text,
    )


def run_suite(
    cases: Sequence[EvalCase],
    *,
    run_provider: Any,
    continue_on_failure: bool = False,
) -> EvalReport:
    """Run a full suite. By default aborts at first failure."""

    results: list[EvalResult] = []
    for case in cases:
        result = run_case(case, run_provider=run_provider)
        results.append(result)
        if not result.passed and not continue_on_failure:
            break
    return EvalReport(results=tuple(results))


def find_failed(report: EvalReport) -> Iterable[EvalResult]:
    """Yield only failed results — useful for diagnostics."""

    return (r for r in report.results if not r.passed)


__all__ = [
    "EvalCase",
    "EvalError",
    "EvalReport",
    "EvalResult",
    "ToolCallExpectation",
    "find_failed",
    "run_case",
    "run_suite",
]
