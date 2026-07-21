"""Policy validation and budget behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soteria_loop.models import TokenUsage
from soteria_loop.policies import LoopPolicy
from soteria_loop.state import StopReason


def test_policy_defaults_are_bounded() -> None:
    policy = LoopPolicy()

    assert policy.max_steps == 20
    assert policy.max_runtime_seconds == 300
    assert policy.repeated_action_limit == 3
    assert policy.consecutive_error_limit == 3
    assert policy.no_progress_window == 5
    assert policy.checkpoint_every_step is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_steps", -1),
        ("max_runtime_seconds", 0),
        ("max_input_tokens", -2),
        ("max_output_tokens", 0),
        ("max_total_tokens", -1),
        ("repeated_action_limit", 0),
        ("consecutive_error_limit", 0),
        ("no_progress_window", -1),
    ],
)
def test_policy_rejects_non_positive_limits(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        LoopPolicy.model_validate({field: value})


def test_policy_accepts_disabled_optional_limits() -> None:
    policy = LoopPolicy(
        max_runtime_seconds=None,
        max_input_tokens=None,
        max_output_tokens=None,
        max_total_tokens=None,
    )

    assert policy.runtime_reason(10_000) is None
    assert (
        policy.token_budget_reason(
            TokenUsage(input_tokens=10_000),
            accounting_available=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("policy", "usage"),
    [
        (LoopPolicy(max_input_tokens=4), TokenUsage(input_tokens=5)),
        (LoopPolicy(max_output_tokens=4), TokenUsage(output_tokens=5)),
        (
            LoopPolicy(max_total_tokens=4),
            TokenUsage(input_tokens=3, output_tokens=2),
        ),
    ],
)
def test_each_token_budget_is_enforced(
    policy: LoopPolicy,
    usage: TokenUsage,
) -> None:
    assert (
        policy.token_budget_reason(usage, accounting_available=True)
        is StopReason.TOKEN_BUDGET_EXCEEDED
    )


def test_token_budget_at_limit_is_allowed() -> None:
    policy = LoopPolicy(max_total_tokens=5)

    assert (
        policy.token_budget_reason(
            TokenUsage(input_tokens=3, output_tokens=2),
            accounting_available=True,
        )
        is None
    )


def test_missing_accounting_is_not_treated_as_zero_usage() -> None:
    policy = LoopPolicy(max_total_tokens=1)

    assert (
        policy.token_budget_reason(
            TokenUsage(input_tokens=50),
            accounting_available=False,
        )
        is None
    )


def test_runtime_limit_triggers_at_boundary() -> None:
    policy = LoopPolicy(max_runtime_seconds=2.5)

    assert policy.runtime_reason(2.49) is None
    assert policy.runtime_reason(2.5) is StopReason.MAX_RUNTIME


def test_policy_is_frozen_and_rejects_unknown_fields() -> None:
    policy = LoopPolicy()
    with pytest.raises(ValidationError):
        policy.max_steps = 3
    with pytest.raises(ValidationError):
        LoopPolicy.model_validate({"unknown": True})
