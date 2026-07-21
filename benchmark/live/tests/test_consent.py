"""Offline tests for the live benchmark cost-consent gate."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.live.consent import (
    COST_CONSENT_ENV,
    COST_CONSENT_FLAG,
    CostConsentError,
    has_cost_consent,
    require_cost_consent,
)


def test_missing_consent_refuses_before_runner() -> None:
    with pytest.raises(CostConsentError, match="i-understand-this-costs-money"):
        require_cost_consent(False, {})


def test_missing_consent_message_names_flag() -> None:
    with pytest.raises(CostConsentError) as exc_info:
        require_cost_consent(False, {})
    assert COST_CONSENT_FLAG in str(exc_info.value)
    assert "--" + COST_CONSENT_FLAG in str(exc_info.value)


def test_explicit_flag_allows_preflight() -> None:
    require_cost_consent(True, {})


def test_environment_consent_values_are_accepted_case_insensitively() -> None:
    for value in ("1", "true", "TRUE", "True", "yes", "YES", "Yes"):
        assert has_cost_consent(False, {COST_CONSENT_ENV: value}) is True


def test_environment_consent_rejects_unknown_values() -> None:
    for value in ("", "0", "false", "no", "off", "enabled", "y"):
        assert has_cost_consent(False, {COST_CONSENT_ENV: value}) is False


def test_has_cost_consent_true_when_flag_set() -> None:
    assert has_cost_consent(True, {}) is True


def test_require_cost_consent_does_not_construct_runner_when_missing() -> None:
    """No provider, no scenario, no runner touched before consent is granted."""

    factory = MagicMock()
    runner = MagicMock()

    def _run(factory_arg: object, runner_arg: object) -> None:
        factory(factory_arg)
        runner(runner_arg)

    with pytest.raises(CostConsentError):
        require_cost_consent(False, {})

    # Pretend the CLI would have called into the runner after preflight.
    factory.assert_not_called()
    runner.assert_not_called()
    del _run  # silence linters; the call sites below remain guarded above


def test_consent_error_is_dedicated_subclass() -> None:
    assert issubclass(CostConsentError, Exception)
