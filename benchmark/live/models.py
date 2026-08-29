"""Public Pydantic models and helpers for the live benchmark."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from hernness.models import HernnessModel, TokenUsage, utc_now
from hernness.state import RunState, StopReason

RawOutcome = Literal["completed", "hit_manual_step_cap", "error"]
Approach = Literal["raw", "hernness"]


class UnexpectedRawLoopError(Exception):
    """Raised when the raw loop encounters an unexpected exception.

    Carries the partial ``LiveRunRecord`` describing the failure so callers can
    persist diagnostics before the exception propagates.
    """

    def __init__(self, record: LiveRunRecord, original: Exception) -> None:
        super().__init__(str(original))
        self.record = record
        self.original = original


class LiveRunRecord(HernnessModel):
    """One observable live benchmark run, agnostic to approach.

    Only public fields are serialized; credential-like keys are not stored on
    the record so JSON dumps cannot leak secrets even if a provider is
    attached by reference.
    """

    model_config = ConfigDict(extra="forbid")

    # Provider identification.
    provider: str | None = None
    api_style: str | None = None
    model: str | None = None

    # Run identification.
    scenario: str = Field(min_length=1)
    approach: Approach
    run_index: int | None = Field(default=None, ge=0)

    # Outcome (raw approach uses ``outcome``; hernness approach uses status/stop_reason).
    outcome: RawOutcome | None = None
    status: RunState | None = None
    stop_reason: StopReason | None = None

    # Metrics.
    steps: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    token_accounting_available: bool = True

    # Safety fences.
    repeated_action_detected: bool | None = None
    manual_step_cap_hit: bool = False
    resume_tool_executed_exactly_once: bool | None = None

    # Error capture.
    expected_error_type: str | None = None
    unexpected_error_type: str | None = None
    unexpected_error_message: str | None = None

    # Optional Hernness trace text.
    trace_text: str | None = None

    @property
    def loop_contained(self) -> bool:
        """The raw loop never reports containment; only Hernness policy stops do."""

        return False


class LiveResults(HernnessModel):
    """A bundle of ``LiveRunRecord`` rows persisted as the source of truth."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    api_style: str | None = None
    model: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=utc_now)
    runs: int = Field(default=0, ge=0)
    records: list[LiveRunRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _runs_matches_records(self) -> LiveResults:
        if self.runs != len(self.records):
            raise ValueError(
                f"runs count {self.runs} does not match records length {len(self.records)}"
            )
        return self

    def write_json(self, output_dir: Path) -> Path:
        """Persist the bundle to ``live_results_<UTC>.json`` under ``output_dir``.

        Returns the absolute path of the written file so the CLI can print it
        in its summary.
        """

        timestamp = self.recorded_at.strftime("%Y%m%dT%H%M%SZ")
        target = Path(output_dir) / f"live_results_{timestamp}.json"
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target


__all__ = [
    "Approach",
    "LiveResults",
    "LiveRunRecord",
    "RawOutcome",
    "UnexpectedRawLoopError",
]
