"""Budget enforcement on top of :class:`UsageTracker`.

A budget caps cumulative spend per run. Two thresholds:

* ``warning_usd`` — emit a notification when crossed
* ``hard_limit_usd`` — block further provider calls when crossed

The checker does not enforce by itself — it is a pure decision function
called by the runtime before each provider call. Keeping it stateless
makes it trivial to test and to compose with retry/backoff helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from hernness.exceptions import HernnessError
from hernness.models import TokenUsage
from hernness.usage import estimate_cost

BudgetError = HernnessError


@dataclass(frozen=True)
class BudgetConfig:
    """Budget envelope for a run."""

    warning_usd: Decimal | None = None
    hard_limit_usd: Decimal | None = None

    def __post_init__(self) -> None:
        if self.warning_usd is not None and self.warning_usd < 0:
            raise BudgetError("warning_usd must be non-negative")
        if self.hard_limit_usd is not None and self.hard_limit_usd < 0:
            raise BudgetError("hard_limit_usd must be non-negative")
        if (
            self.warning_usd is not None
            and self.hard_limit_usd is not None
            and self.warning_usd > self.hard_limit_usd
        ):
            raise BudgetError("warning_usd must not exceed hard_limit_usd")

    @property
    def has_limits(self) -> bool:
        return self.warning_usd is not None or self.hard_limit_usd is not None


@dataclass(frozen=True)
class BudgetDecision:
    """Result of :meth:`BudgetChecker.check`."""

    allowed: bool
    spent_usd: Decimal | None
    crossed_warning: bool
    exceeded_hard_limit: bool


class BudgetChecker:
    """Stateless budget gate — call :meth:`check` per provider call."""

    __slots__ = ("_config",)

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config = config or BudgetConfig()

    @property
    def config(self) -> BudgetConfig:
        return self._config

    def check(
        self, total: TokenUsage, *, rates: tuple[Decimal, Decimal] | None = None
    ) -> BudgetDecision:
        spent = estimate_cost(total, rates=rates)
        if spent is None:
            return BudgetDecision(
                allowed=True,
                spent_usd=None,
                crossed_warning=False,
                exceeded_hard_limit=False,
            )
        warning = self._config.warning_usd
        hard = self._config.hard_limit_usd
        crossed_warning = warning is not None and spent >= warning
        exceeded = hard is not None and spent >= hard
        return BudgetDecision(
            allowed=not exceeded,
            spent_usd=spent,
            crossed_warning=crossed_warning,
            exceeded_hard_limit=exceeded,
        )


__all__ = [
    "BudgetChecker",
    "BudgetConfig",
    "BudgetDecision",
    "BudgetError",
]
