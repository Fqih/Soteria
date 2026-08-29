"""Sub-agent delegation.

A :class:`SubAgentRunner` spawns a child run scoped under a parent
``run_id``. The child has its own checkpoint namespace (parent.child)
and can record checkpoints / run history independently. The runner
returns a :class:`SubAgentResult` carrying the child's outputs and a
reference back to its parent.

The runner is intentionally minimal — it does not call a real provider.
Callers supply a ``run_step`` callable that executes one logical step
against their own provider / tool stack. This keeps the module testable
without external dependencies and easy to wire into an AgentRuntime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from hernness.exceptions import HernnessError
from hernness.snapshot import resume

SubAgentError = HernnessError


@dataclass(frozen=True)
class SubAgentResult:
    """Outcome of one delegated sub-agent run."""

    parent_run_id: str
    child_run_id: str
    steps: int
    outputs: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def child_run_id(parent: str, child: str) -> str:
    """Build a deterministic ``parent.child`` namespace."""

    if not parent:
        raise SubAgentError("parent_run_id must be non-empty")
    if not child:
        raise SubAgentError("child_name must be non-empty")
    return f"{parent}.{child}"


@dataclass(frozen=True)
class SubAgentRunner:
    """Spawn child runs against a parent-scoped checkpoint store."""

    run_step: Callable[[str, int], Any]

    def run(
        self,
        parent_run_id: str,
        child_name: str,
        *,
        max_steps: int = 1,
        resume_from: Any = None,
    ) -> SubAgentResult:
        if max_steps < 1:
            raise SubAgentError("max_steps must be >= 1")
        cid = child_run_id(parent_run_id, child_name)
        start_step = 0
        if resume_from is not None:
            plan = resume(resume_from, cid)
            if plan is not None:
                start_step = plan.step + 1
        outputs: list[Any] = []
        for step in range(start_step, start_step + max_steps):
            outputs.append(self.run_step(cid, step))
        return SubAgentResult(
            parent_run_id=parent_run_id,
            child_run_id=cid,
            steps=max_steps,
            outputs=tuple(outputs),
        )

    def run_many(
        self,
        parent_run_id: str,
        children: Iterable[tuple[str, dict[str, Any]]],
    ) -> tuple[SubAgentResult, ...]:
        return tuple(self.run(parent_run_id, name, **(opts or {})) for name, opts in children)


__all__ = [
    "SubAgentError",
    "SubAgentResult",
    "SubAgentRunner",
    "child_run_id",
]
