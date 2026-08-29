"""Deterministic repetition and no-progress heuristics."""

from __future__ import annotations

from pydantic import JsonValue

from hernness.models import ModelResponse, ToolCall, ToolResult
from hernness.tools import (
    canonical_fingerprint,
    tool_call_fingerprint,
    tool_result_fingerprint,
)


def model_response_fingerprint(response: ModelResponse) -> str:
    """Fingerprint stable response content while excluding IDs and token usage."""

    if response.tool_call is not None:
        value: JsonValue = {
            "tool": {
                "name": response.tool_call.name,
                "arguments": response.tool_call.arguments,
            }
        }
    else:
        value = {"content": response.content}
    return canonical_fingerprint(value)


class ProgressDetector:
    """Track deterministic fingerprints used by reliability policies.

    These checks identify exact repetition. They do not claim to detect
    semantic equivalence or prove that useful progress is impossible.
    """

    def __init__(
        self,
        *,
        action_history: list[str] | None = None,
        observation_history: list[str] | None = None,
        model_history: list[str] | None = None,
        progress_markers: list[str] | None = None,
    ) -> None:
        self.action_history = list(action_history or [])
        self.observation_history = list(observation_history or [])
        self.model_history = list(model_history or [])
        self.progress_markers = list(progress_markers or [])

    def record_action(self, call: ToolCall) -> str:
        """Record and return the normalized tool-call fingerprint."""

        fingerprint = tool_call_fingerprint(call)
        self.action_history.append(fingerprint)
        return fingerprint

    def record_observation(self, result: ToolResult) -> str:
        """Record and return a normalized tool-result fingerprint."""

        fingerprint = tool_result_fingerprint(result)
        self.observation_history.append(fingerprint)
        return fingerprint

    def record_model_response(self, response: ModelResponse) -> str:
        """Record and return a normalized model-response fingerprint."""

        fingerprint = model_response_fingerprint(response)
        self.model_history.append(fingerprint)
        return fingerprint

    def record_progress_marker(self, marker: JsonValue) -> str:
        """Record an application-defined, JSON-safe progress marker."""

        fingerprint = canonical_fingerprint(marker)
        self.progress_markers.append(fingerprint)
        return fingerprint

    def repeated_action(self, limit: int) -> bool:
        """Return true when the latest action appears limit times consecutively."""

        return self._tail_is_identical(self.action_history, limit)

    def no_progress(self, window: int) -> bool:
        """Return true for an unchanged observation, response, or explicit marker window."""

        return any(
            self._tail_is_identical(history, window)
            for history in (
                self.observation_history,
                self.model_history,
                self.progress_markers,
            )
        )

    @staticmethod
    def _tail_is_identical(history: list[str], size: int) -> bool:
        if size <= 0:
            raise ValueError("fingerprint window size must be positive")
        if len(history) < size:
            return False
        tail = history[-size:]
        return len(set(tail)) == 1
