"""Hook registry for the Avo runtime.

Hooks are pluggable observers (and one veto point) that the runtime
fires at well-defined moments:

* ``PreToolUse`` — fired in ``handle_approval_pending`` *before* the
  operator prompt. Returning ``HookDecision(action=BLOCK)`` denies the
  tool call without prompting the operator; ``action=ALLOW`` continues
  the normal approval flow.
* ``PostToolUse`` — fired after ``TOOL_COMPLETED`` (or ``TOOL_FAILED``)
  is persisted. Always informational; the tool has already run.
* ``Stop`` — fired once per run when it transitions to a terminal
  state. Informational; the run is already done.
* ``Notification`` — fired for internal events such as
  ``consecutive_errors`` reaching the limit. Informational.

Hooks return ``HookDecision | None``. ``None`` is treated as ``ALLOW``
so a hook that has nothing to say stays silent.

The registry stores hooks in registration order and fires them
sequentially. The first blocking decision short-circuits the rest, so
a PreToolUse chain stops at the first veto.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from pydantic import JsonValue

from avo.models import RunRecord, ToolCall, ToolResult


class HookEvent(StrEnum):
    """The hook firing points exposed to operator code."""

    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STOP = "stop"
    NOTIFICATION = "notification"


class HookAction(StrEnum):
    """The decision a hook can return for ``PRE_TOOL_USE``."""

    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True)
class HookDecision:
    """The outcome a hook returns to the runtime."""

    action: HookAction = HookAction.ALLOW
    reason: str = ""
    modified_args: dict[str, JsonValue] | None = None

    @classmethod
    def allow(cls) -> HookDecision:
        return cls(action=HookAction.ALLOW)

    @classmethod
    def block(cls, reason: str) -> HookDecision:
        return cls(action=HookAction.BLOCK, reason=reason)


@dataclass(frozen=True)
class HookContext:
    """The payload passed to every hook invocation."""

    event: HookEvent
    run_id: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    run: RunRecord | None = None
    notification: str = ""
    extra: dict[str, JsonValue] = field(default_factory=dict)


HookCallable = Callable[[HookContext], HookDecision | Awaitable[HookDecision]]


def _is_awaitable(value: object) -> bool:
    """Return whether ``value`` is awaitable without importing asyncio."""

    return hasattr(value, "__await__")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class HookRegistry:
    """A named, ordered collection of hooks keyed by ``HookEvent``."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallable]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, hook: HookCallable) -> None:
        """Append ``hook`` to the chain for ``event``."""

        self._hooks[event].append(hook)

    def hooks_for(self, event: HookEvent) -> list[HookCallable]:
        """Return a copy of the hooks registered for ``event``."""

        return list(self._hooks[event])

    async def fire(self, context: HookContext) -> HookDecision:
        """Fire the chain for ``context.event``.

        Returns the first ``BLOCK`` decision, or the final ``ALLOW``.
        Hooks raising are treated as ``ALLOW`` with the exception
        captured in ``extra`` so a single broken hook cannot wedge the
        runtime.
        """

        chain = self._hooks[context.event]
        if not chain:
            return HookDecision.allow()
        final = HookDecision.allow()
        for hook in chain:
            try:
                value = hook(context)
                if _is_awaitable(value):
                    awaited = cast(Awaitable[HookDecision], value)
                    value = await awaited
                decision = value if isinstance(value, HookDecision) else HookDecision.allow()
            except Exception as exc:
                decision = HookDecision(
                    action=HookAction.ALLOW,
                    reason=f"hook raised {type(exc).__name__}: {exc}",
                )
            if decision.action is HookAction.BLOCK:
                return decision
            final = decision
        return final


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def make_blocking_hook(reason: str) -> HookCallable:
    """Return a hook that always blocks ``PRE_TOOL_USE`` for a named tool."""

    async def hook(context: HookContext) -> HookDecision:
        if context.event is not HookEvent.PRE_TOOL_USE:
            return HookDecision.allow()
        return HookDecision.block(reason)

    return hook


def make_logging_hook(sink: list[HookContext]) -> HookCallable:
    """Return a hook that records every invocation into ``sink``."""

    async def hook(context: HookContext) -> HookDecision:
        sink.append(context)
        return HookDecision.allow()

    return hook


__all__ = [
    "HookAction",
    "HookCallable",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookRegistry",
    "make_blocking_hook",
    "make_logging_hook",
]
