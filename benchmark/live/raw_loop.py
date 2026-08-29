"""Minimal provider-and-tool loop used as the raw baseline for the live benchmark.

This loop intentionally avoids ``AgentRuntime``, ``LoopPolicy``,
``ProgressDetector``, checkpoints, and the repeated-action stop. It exercises
only ``ModelProvider.generate`` and ``ToolRegistry.invoke`` so the recorded
metrics describe the raw provider behavior, not any Hernness safety net.
"""

from __future__ import annotations

import time
from typing import cast

from pydantic import JsonValue

from benchmark.live.models import (
    LiveRunRecord,
    RawOutcome,
    UnexpectedRawLoopError,
)
from benchmark.live.scenarios import LiveScenario
from hernness.exceptions import ProviderError, ToolExecutionError
from hernness.models import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from hernness.providers.base import ModelProvider
from hernness.tools import ToolRegistry

_RAW_RUN_ID = "raw-loop"


def classify_raw_outcome(record: LiveRunRecord) -> RawOutcome:
    """Return the canonical raw-loop outcome label for a record.

    Prefers an explicit ``outcome`` if one is set; otherwise falls back to the
    manual-cap and error fields so the classifier is robust to records built
    out of order.
    """

    if record.outcome == "completed":
        return "completed"
    if record.manual_step_cap_hit:
        return "hit_manual_step_cap"
    return "error"


async def run_raw_loop(
    provider: ModelProvider,
    scenario: LiveScenario,
    manual_step_cap: int,
    max_completion_tokens: int,
) -> LiveRunRecord:
    """Drive the raw provider-and-tool loop until it terminates.

    Args:
        provider: The provider to call for every model decision.
        scenario: The live scenario whose task, tool set, and policy metadata
            describe the run.
        manual_step_cap: External safety fence; the loop exits after this many
            model calls regardless of the model's behavior.
        max_completion_tokens: Provider hint surfaced only via the
            ``ModelRequest`` to keep parity with the Hernness runner.

    Returns:
        A ``LiveRunRecord`` describing the run. ``loop_contained`` is always
        ``False`` because the raw loop has no policy-driven containment.

    Raises:
        UnexpectedRawLoopError: When an exception outside ``ProviderError`` and
            ``ToolExecutionError`` escapes the loop body. The attached record
            captures the partial state for diagnostics.
    """

    if manual_step_cap < 1:
        raise ValueError("manual_step_cap must be at least 1")

    # ``max_completion_tokens`` is intentionally not threaded through the raw
    # loop's ``ModelRequest``; providers consume it via their own configuration
    # and the parameter is retained so the public signature matches the
    # Hernness runner. Touching it once suppresses the unused-argument warning
    # without altering the loop body.
    _ = max_completion_tokens

    tools = ToolRegistry(scenario.tools())
    messages: list[dict[str, JsonValue]] = [{"role": "user", "content": scenario.task}]
    # The raw loop never persists a completed-tool-call set because it has no
    # resume safety net; passing an empty set to ``ToolRegistry.invoke`` keeps
    # the registry's idempotency check permissive.
    completed_tool_call_ids: set[str] = set()
    steps = 0
    token_usage = TokenUsage()
    token_accounting_available = True
    started = time.perf_counter()
    tool_metadata = tools.metadata

    def _build(
        *,
        outcome: RawOutcome,
        manual_step_cap_hit: bool = False,
        expected_error_type: str | None = None,
        unexpected_error_type: str | None = None,
        unexpected_error_message: str | None = None,
    ) -> LiveRunRecord:
        return LiveRunRecord(
            scenario=scenario.name,
            approach="raw",
            outcome=outcome,
            steps=steps,
            duration_seconds=max(0.0, time.perf_counter() - started),
            token_usage=token_usage,
            token_accounting_available=token_accounting_available,
            manual_step_cap_hit=manual_step_cap_hit,
            expected_error_type=expected_error_type,
            unexpected_error_type=unexpected_error_type,
            unexpected_error_message=unexpected_error_message,
        )

    try:
        while True:
            steps += 1
            request = ModelRequest(
                run_id=_RAW_RUN_ID,
                step=steps,
                messages=messages,
                tools=tool_metadata,
            )

            try:
                response = await provider.generate(request)
            except ProviderError:
                return _build(outcome="error", expected_error_type="ProviderError")
            except ToolExecutionError:
                return _build(outcome="error", expected_error_type="ToolExecutionError")

            if response.usage is None:
                token_accounting_available = False
            else:
                token_usage = token_usage.plus(response.usage)

            messages.append(_assistant_message(response))

            if response.is_final:
                return _build(outcome="completed")

            call = _require_tool_call(response)
            # The raw loop never tracks completed tool-call IDs because it has
            # no resume safety net; a live provider may legitimately reuse an
            # identifier when scripted with ``repeat_last``.
            result = await tools.invoke(call, completed_tool_call_ids=completed_tool_call_ids)
            messages.append(_tool_message(result))

            if not result.success:
                return _build(
                    outcome="error",
                    expected_error_type=_classify_tool_error(result.error),
                )

            if steps >= manual_step_cap:
                return _build(outcome="hit_manual_step_cap", manual_step_cap_hit=True)
    except UnexpectedRawLoopError:
        raise
    except Exception as exc:
        record = _build(
            outcome="error",
            unexpected_error_type=type(exc).__name__,
            unexpected_error_message=str(exc),
        )
        raise UnexpectedRawLoopError(record, exc) from exc


def _classify_tool_error(error: str | None) -> str:
    """Infer the originating exception class from a failed ``ToolResult``."""

    if not error:
        return "ToolError"
    if "is not registered" in error:
        return "ToolNotFoundError"
    if "Invalid arguments" in error:
        return "ToolValidationError"
    if "raised" in error:
        return "ToolExecutionError"
    return "ToolError"


def _assistant_message(response: ModelResponse) -> dict[str, JsonValue]:
    if response.tool_call is not None:
        return {
            "role": "assistant",
            "tool_call": cast(JsonValue, response.tool_call.model_dump(mode="json")),
        }
    return {"role": "assistant", "content": response.content}


def _tool_message(result: ToolResult) -> dict[str, JsonValue]:
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": cast(JsonValue, result.model_dump(mode="json")),
    }


def _require_tool_call(response: ModelResponse) -> ToolCall:
    if response.tool_call is None:
        raise RuntimeError("Raw loop received a non-final response without a tool call.")
    return response.tool_call


__all__ = ["classify_raw_outcome", "run_raw_loop"]
