"""Cost-consent gate for the opt-in live benchmark CLI.

The live benchmark spends real money against paid APIs.  Before any provider
construction, scenario orchestration, or HTTP traffic, the CLI must collect an
explicit, auditable opt-in.  Two equivalent channels are accepted:

* The ``--i-understand-this-costs-money`` CLI flag (``flag``).
* The ``HERNNESS_I_UNDERSTAND_THIS_COSTS_MONEY`` environment variable, which
  must hold one of ``1``, ``true``, or ``yes`` (case-insensitive).

The module exposes pure helpers that take the consent inputs explicitly so the
CLI can drive preflight and tests can exercise every branch without touching
the real environment.
"""

from __future__ import annotations

from collections.abc import Mapping

COST_CONSENT_FLAG = "i-understand-this-costs-money"
COST_CONSENT_ENV = "HERNNESS_I_UNDERSTAND_THIS_COSTS_MONEY"
_TRUTHY_VALUES = frozenset({"1", "true", "yes"})


class CostConsentError(Exception):
    """Raised when the live benchmark is invoked without explicit cost consent."""


def has_cost_consent(flag: bool, environ: Mapping[str, str]) -> bool:
    """Return True when either the CLI flag or env var grants consent.

    Args:
        flag: Value of ``--i-understand-this-costs-money``.
        environ: Mapping consulted for ``HERNNESS_I_UNDERSTAND_THIS_COSTS_MONEY``.

    Returns:
        ``True`` when consent is granted via either channel.
    """

    if flag:
        return True
    value = environ.get(COST_CONSENT_ENV)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def require_cost_consent(flag: bool, environ: Mapping[str, str]) -> None:
    """Raise :class:`CostConsentError` when no consent channel is set.

    The error message names the CLI flag verbatim so first-time users can find
    the right knob without consulting the docs.
    """

    if has_cost_consent(flag, environ):
        return
    raise CostConsentError(
        "Live benchmark refused: pass --"
        f"{COST_CONSENT_FLAG} or set {COST_CONSENT_ENV}=1|true|yes before running."
    )


__all__ = [
    "COST_CONSENT_ENV",
    "COST_CONSENT_FLAG",
    "CostConsentError",
    "has_cost_consent",
    "require_cost_consent",
]
