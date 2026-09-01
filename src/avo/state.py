"""Run states, stop reasons, and central transition validation."""

from __future__ import annotations

from enum import StrEnum

from avo.exceptions import InvalidStateTransitionError


class RunState(StrEnum):
    """The durable execution states in the Avo state machine."""

    CREATED = "created"
    MODEL_PENDING = "model_pending"
    DECISION_RECEIVED = "decision_received"
    TOOL_PENDING = "tool_pending"
    APPROVAL_PENDING = "approval_pending"
    TOOL_EXECUTING = "tool_executing"
    OBSERVATION_RECORDED = "observation_recorded"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELLED = "cancelled"


class StopReason(StrEnum):
    """The explicit reason a terminal run stopped."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_RUNTIME = "max_runtime"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    REPEATED_ACTION = "repeated_action"
    NO_PROGRESS = "no_progress"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    POLICY_DENIED = "policy_denied"
    USER_CANCELLED = "user_cancelled"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    INTERNAL_ERROR = "internal_error"


TERMINAL_STATES = frozenset(
    {RunState.COMPLETED, RunState.FAILED, RunState.STOPPED, RunState.CANCELLED}
)

_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset(
        {RunState.MODEL_PENDING, RunState.PAUSED, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.MODEL_PENDING: frozenset(
        {
            RunState.DECISION_RECEIVED,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.DECISION_RECEIVED: frozenset(
        {
            RunState.TOOL_PENDING,
            RunState.COMPLETED,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.TOOL_PENDING: frozenset(
        {
            RunState.APPROVAL_PENDING,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.APPROVAL_PENDING: frozenset(
        {
            RunState.TOOL_EXECUTING,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.TOOL_EXECUTING: frozenset(
        {
            RunState.OBSERVATION_RECORDED,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.OBSERVATION_RECORDED: frozenset(
        {
            RunState.MODEL_PENDING,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.PAUSED: frozenset(
        {
            RunState.MODEL_PENDING,
            RunState.DECISION_RECEIVED,
            RunState.TOOL_PENDING,
            RunState.APPROVAL_PENDING,
            RunState.TOOL_EXECUTING,
            RunState.OBSERVATION_RECORDED,
            RunState.FAILED,
            RunState.STOPPED,
            RunState.CANCELLED,
        }
    ),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.STOPPED: frozenset(),
    RunState.CANCELLED: frozenset(),
}

_STOP_REASONS_BY_STATE: dict[RunState, frozenset[StopReason]] = {
    RunState.COMPLETED: frozenset({StopReason.COMPLETED}),
    RunState.STOPPED: frozenset(
        {
            StopReason.MAX_STEPS,
            StopReason.MAX_RUNTIME,
            StopReason.TOKEN_BUDGET_EXCEEDED,
            StopReason.REPEATED_ACTION,
            StopReason.NO_PROGRESS,
            StopReason.CONSECUTIVE_ERRORS,
            StopReason.POLICY_DENIED,
        }
    ),
    RunState.FAILED: frozenset(
        {
            StopReason.PROVIDER_ERROR,
            StopReason.TOOL_ERROR,
            StopReason.INVALID_MODEL_RESPONSE,
            StopReason.INTERNAL_ERROR,
        }
    ),
    RunState.CANCELLED: frozenset({StopReason.USER_CANCELLED}),
}


def is_terminal(state: RunState) -> bool:
    """Return whether state is terminal."""

    return state in TERMINAL_STATES


def validate_transition(current: RunState, target: RunState) -> None:
    """Validate a state transition.

    Raises:
        InvalidStateTransitionError: If the target cannot follow the current state.
    """

    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidStateTransitionError(
            f"Cannot transition run from {current.value!r} to {target.value!r}."
        )


def validate_terminal_outcome(state: RunState, reason: StopReason) -> None:
    """Validate that a terminal state agrees with its stop reason."""

    allowed = _STOP_REASONS_BY_STATE.get(state)
    if allowed is None:
        raise InvalidStateTransitionError(
            f"State {state.value!r} is not terminal and cannot have stop reason {reason.value!r}."
        )
    if reason not in allowed:
        expected = ", ".join(sorted(item.value for item in allowed))
        raise InvalidStateTransitionError(
            f"Stop reason {reason.value!r} is invalid for state {state.value!r}; "
            f"expected one of: {expected}."
        )
