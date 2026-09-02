"""Validated reliability limits and deterministic policy decisions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from avo.circuit_breaker import CircuitBreakerPolicy
from avo.models import TokenUsage
from avo.state import StopReason


class LoopPolicy(BaseModel):
    """Runtime limits enforced at deterministic operation boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=20, gt=0)
    max_runtime_seconds: float | None = Field(default=300, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, gt=0)
    repeated_action_limit: int = Field(default=3, gt=0)
    consecutive_error_limit: int = Field(default=3, gt=0)
    no_progress_window: int = Field(default=5, gt=0)
    checkpoint_every_step: bool = True
    provider_timeout_seconds: float | None = Field(default=60, gt=0)
    tool_timeout_seconds: float | None = Field(default=60, gt=0)
    circuit_breaker: CircuitBreakerPolicy | None = Field(default=None)

    def token_budget_reason(
        self,
        usage: TokenUsage,
        *,
        accounting_available: bool,
    ) -> StopReason | None:
        """Return a token-budget stop reason when reported usage exceeds a limit.

        Missing provider usage does not invent a zero count. The runtime records
        accounting_available=False and cannot enforce a numeric token budget.
        """

        if not accounting_available:
            return None
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            return StopReason.TOKEN_BUDGET_EXCEEDED
        if self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            return StopReason.TOKEN_BUDGET_EXCEEDED
        if self.max_total_tokens is not None and usage.total_tokens > self.max_total_tokens:
            return StopReason.TOKEN_BUDGET_EXCEEDED
        return None

    def runtime_reason(self, elapsed_seconds: float) -> StopReason | None:
        """Return a runtime-budget stop reason at an operation boundary."""

        if self.max_runtime_seconds is not None and elapsed_seconds >= self.max_runtime_seconds:
            return StopReason.MAX_RUNTIME
        return None
