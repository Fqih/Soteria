"""Tests for the eval framework."""

from __future__ import annotations

from typing import Any

import pytest

from soteria_loop.eval import (
    EvalCase,
    EvalError,
    EvalReport,
    EvalResult,
    ToolCallExpectation,
    find_failed,
    run_case,
    run_suite,
)


def _static_provider(text: str, tool_calls: list[dict[str, Any]] | None = None) -> Any:
    def _run(prompt: str) -> dict[str, Any]:
        return {"content": text, "tool_calls": list(tool_calls or [])}

    return _run


def test_run_case_passes_when_text_matches() -> None:
    case = EvalCase(name="greet", prompt="hi", expected_text_substrings=("hello",))
    result = run_case(case, run_provider=_static_provider("hello world"))
    assert result.passed
    assert result.failures == ()


def test_run_case_fails_when_text_missing() -> None:
    case = EvalCase(name="greet", prompt="hi", expected_text_substrings=("hello",))
    result = run_case(case, run_provider=_static_provider("goodbye"))
    assert not result.passed
    assert any("hello" in f for f in result.failures)


def test_run_case_fails_when_tool_call_missing() -> None:
    case = EvalCase(
        name="search",
        prompt="find",
        expected_tool_calls=(ToolCallExpectation(name="search"),),
    )
    result = run_case(case, run_provider=_static_provider("ok", tool_calls=[]))
    assert not result.passed
    assert any("search" in f for f in result.failures)


def test_run_case_validates_tool_call_arguments() -> None:
    case = EvalCase(
        name="search",
        prompt="find",
        expected_tool_calls=(
            ToolCallExpectation(name="search", arguments_subset={"query": "cats"}),
        ),
    )
    provider = _static_provider(
        "ok", tool_calls=[{"name": "search", "arguments": {"query": "dogs"}}]
    )
    result = run_case(case, run_provider=provider)
    assert not result.passed
    assert any("query" in f and "cats" in f for f in result.failures)


def test_run_case_passes_when_tool_call_arguments_match() -> None:
    case = EvalCase(
        name="search",
        prompt="find",
        expected_tool_calls=(
            ToolCallExpectation(name="search", arguments_subset={"query": "cats"}),
        ),
    )
    provider = _static_provider(
        "ok", tool_calls=[{"name": "search", "arguments": {"query": "cats", "limit": 5}}]
    )
    result = run_case(case, run_provider=provider)
    assert result.passed


def test_run_case_records_provider_errors() -> None:
    case = EvalCase(name="x", prompt="y")

    def broken(_prompt: str) -> dict[str, Any]:
        raise RuntimeError("kaboom")

    result = run_case(case, run_provider=broken)
    assert not result.passed
    assert "RuntimeError" in result.failures[0]


def test_run_suite_aborts_on_first_failure() -> None:
    cases = (
        EvalCase(name="a", prompt="x", expected_text_substrings=("alpha",)),
        EvalCase(name="b", prompt="y", expected_text_substrings=("beta",)),
    )

    def provider(prompt: str) -> dict[str, Any]:
        return {"content": prompt.upper(), "tool_calls": []}

    report = run_suite(cases, run_provider=provider)
    assert report.total == 1
    assert report.failed == 1


def test_run_suite_continues_when_requested() -> None:
    cases = (
        EvalCase(name="a", prompt="x", expected_text_substrings=("X",)),
        EvalCase(name="b", prompt="y", expected_text_substrings=("Y",)),
    )

    def provider(prompt: str) -> dict[str, Any]:
        return {"content": prompt.upper(), "tool_calls": []}

    report = run_suite(cases, run_provider=provider, continue_on_failure=True)
    assert report.total == 2
    assert report.passed == 2
    assert report.ok


def test_eval_report_ok_flag() -> None:
    case = EvalCase(name="a", prompt="x")
    result = run_case(case, run_provider=_static_provider("anything"))
    report = EvalReport(results=(result,))
    assert report.ok
    assert report.passed == 1
    assert report.failed == 0


def test_find_failed_returns_only_failures() -> None:
    a = EvalResult(case=EvalCase(name="a", prompt="x"), passed=True)
    b = EvalResult(case=EvalCase(name="b", prompt="x"), passed=False, failures=("z",))
    report = EvalReport(results=(a, b))
    failed = list(find_failed(report))
    assert len(failed) == 1
    assert failed[0].case.name == "b"


def test_eval_error_alias() -> None:
    assert EvalError is not None


def test_eval_case_default_substrings_empty() -> None:
    case = EvalCase(name="x", prompt="y")
    assert case.expected_text_substrings == ()
    assert case.expected_tool_calls == ()
    assert case.tags == ()


def test_tool_call_expectation_default_subset() -> None:
    expectation = ToolCallExpectation(name="search")
    assert expectation.arguments_subset == {}
