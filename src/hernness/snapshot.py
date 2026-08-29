"""Resume-from-checkpoint helpers.

A snapshot bundles the latest :class:`Checkpoint` for a run with a
validated view of the state dict so resume code can rely on typed
access. The validator is intentionally permissive — unknown keys are
preserved but flagged so callers can decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hernness.checkpoint import Checkpoint, CheckpointStore
from hernness.exceptions import HernnessError

ResumeError = HernnessError

_REQUIRED_KEYS: frozenset[str] = frozenset({"step", "phase"})


@dataclass(frozen=True)
class ResumePlan:
    """A typed view of the latest checkpoint, ready for resumption."""

    run_id: str
    sequence: int
    step: int
    phase: str
    state: dict[str, Any]
    extra_keys: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> ResumePlan:
        state = dict(checkpoint.state)
        missing = _REQUIRED_KEYS - state.keys()
        if missing:
            raise ResumeError(
                f"checkpoint {checkpoint.run_id}#{checkpoint.sequence} missing "
                f"required keys: {sorted(missing)}"
            )
        step = state["step"]
        phase = state["phase"]
        if not isinstance(step, int):
            raise ResumeError(f"step must be int, got {type(step).__name__}")
        if not isinstance(phase, str):
            raise ResumeError(f"phase must be str, got {type(phase).__name__}")
        known = _REQUIRED_KEYS
        extras = tuple(k for k in state if k not in known)
        return cls(
            run_id=checkpoint.run_id,
            sequence=checkpoint.sequence,
            step=step,
            phase=phase,
            state=state,
            extra_keys=extras,
        )


def resume(store: CheckpointStore, run_id: str) -> ResumePlan | None:
    """Return a :class:`ResumePlan` for ``run_id`` or ``None`` if no checkpoint."""

    latest = store.latest(run_id)
    if latest is None:
        return None
    return ResumePlan.from_checkpoint(latest)


def require_resume(store: CheckpointStore, run_id: str) -> ResumePlan:
    """Like :func:`resume` but raises when no checkpoint exists."""

    plan = resume(store, run_id)
    if plan is None:
        raise ResumeError(f"no checkpoint found for run {run_id!r}")
    return plan


__all__ = [
    "ResumeError",
    "ResumePlan",
    "require_resume",
    "resume",
]
