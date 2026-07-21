"""Exact fingerprint and no-progress heuristic behavior."""

from __future__ import annotations

from soteria_loop.models import ModelResponse, ToolCall, ToolResult
from soteria_loop.progress import ProgressDetector, model_response_fingerprint
from soteria_loop.tools import canonical_fingerprint, tool_call_fingerprint


def test_canonical_fingerprint_ignores_mapping_order() -> None:
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint({"b": 2, "a": 1})


def test_tool_call_fingerprint_excludes_identifier() -> None:
    first = ToolCall(tool_call_id="one", name="lookup", arguments={"query": "x"})
    second = ToolCall(tool_call_id="two", name="lookup", arguments={"query": "x"})

    assert tool_call_fingerprint(first) == tool_call_fingerprint(second)


def test_model_response_fingerprint_excludes_response_and_call_ids() -> None:
    first = ModelResponse(
        response_id="response-1",
        tool_call=ToolCall(tool_call_id="call-1", name="lookup", arguments={"q": 1}),
    )
    second = ModelResponse(
        response_id="response-2",
        tool_call=ToolCall(tool_call_id="call-2", name="lookup", arguments={"q": 1}),
    )

    assert model_response_fingerprint(first) == model_response_fingerprint(second)


def test_repeated_action_requires_consecutive_tail() -> None:
    detector = ProgressDetector()
    repeated = ToolCall(name="lookup", arguments={"q": "same"})
    different = ToolCall(name="lookup", arguments={"q": "different"})

    detector.record_action(repeated)
    detector.record_action(different)
    detector.record_action(repeated)
    assert detector.repeated_action(2) is False
    detector.record_action(repeated)
    assert detector.repeated_action(2) is True


def test_identical_observations_trigger_no_progress() -> None:
    detector = ProgressDetector()
    for call_id in ("one", "two", "three"):
        detector.record_observation(
            ToolResult(
                tool_call_id=call_id,
                tool_name="lookup",
                success=True,
                output={"same": True},
            )
        )

    assert detector.no_progress(3) is True


def test_different_observations_are_progress() -> None:
    detector = ProgressDetector()
    for value in (1, 2, 3):
        detector.record_observation(
            ToolResult(
                tool_call_id=str(value),
                tool_name="lookup",
                success=True,
                output={"value": value},
            )
        )

    assert detector.no_progress(3) is False


def test_model_and_user_markers_are_deterministic_progress_signals() -> None:
    model_detector = ProgressDetector()
    marker_detector = ProgressDetector()
    for _ in range(2):
        model_detector.record_model_response(ModelResponse(content="unchanged"))
        marker_detector.record_progress_marker({"revision": 1})

    assert model_detector.no_progress(2) is True
    assert marker_detector.no_progress(2) is True


def test_empty_or_short_history_does_not_trigger() -> None:
    detector = ProgressDetector()

    assert detector.repeated_action(2) is False
    assert detector.no_progress(2) is False


def test_non_positive_detector_windows_are_rejected() -> None:
    detector = ProgressDetector()

    try:
        detector.no_progress(0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected no_progress(0) to fail")
