"""Tests for the hook registry and runtime hook firing points."""

from __future__ import annotations

import asyncio

from avo.hooks import (
    HookAction,
    HookContext,
    HookDecision,
    HookEvent,
    HookRegistry,
    make_blocking_hook,
    make_logging_hook,
)
from avo.models import ModelResponse, ToolCall
from avo.policies import LoopPolicy
from avo.providers.fake import FakeProvider
from avo.runtime import AgentRuntime
from avo.storage.memory import InMemoryEventStore


def _runtime_with_hooks(
    hooks: HookRegistry,
    *,
    policy: LoopPolicy,
) -> AgentRuntime:
    """Build an AgentRuntime with a fake provider that always finishes."""

    return AgentRuntime(
        provider=FakeProvider(responses=[]),
        event_store=InMemoryEventStore(),
        policy=policy,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------


async def test_registry_with_no_hooks_allows() -> None:
    registry = HookRegistry()
    decision = await registry.fire(HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x"))
    assert decision.action is HookAction.ALLOW


async def test_registry_allow_decision_returns_allow() -> None:
    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, make_logging_hook([]))
    decision = await registry.fire(HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x"))
    assert decision.action is HookAction.ALLOW


async def test_registry_block_short_circuits() -> None:
    calls: list[str] = []

    async def first(_context: HookContext) -> HookDecision:
        calls.append("first")
        return HookDecision.block("nope")

    async def second(_context: HookContext) -> HookDecision:
        calls.append("second")
        return HookDecision.allow()

    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, first)
    registry.register(HookEvent.PRE_TOOL_USE, second)
    decision = await registry.fire(HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x"))
    assert decision.action is HookAction.BLOCK
    assert calls == ["first"]


async def test_registry_hook_exception_does_not_break_run() -> None:
    calls: list[str] = []

    async def broken(_context: HookContext) -> HookDecision:
        calls.append("broken")
        raise RuntimeError("boom")

    async def after(_context: HookContext) -> HookDecision:
        calls.append("after")
        return HookDecision.allow()

    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, broken)
    registry.register(HookEvent.PRE_TOOL_USE, after)
    decision = await registry.fire(HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x"))
    assert decision.action is HookAction.ALLOW
    assert calls == ["broken", "after"]


async def test_registry_accepts_sync_hook() -> None:
    def sync_hook(_context: HookContext) -> HookDecision:
        return HookDecision.block("sync")

    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, sync_hook)
    decision = await registry.fire(HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x"))
    assert decision.action is HookAction.BLOCK
    assert decision.reason == "sync"


async def test_registry_only_fires_matching_event() -> None:
    sink: list[HookContext] = []
    registry = HookRegistry()
    registry.register(HookEvent.POST_TOOL_USE, make_logging_hook(sink))
    await registry.fire(HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x"))
    assert sink == []


def test_blocking_hook_factory_blocks_only_pre_tool_use() -> None:
    async def run() -> None:
        hook = make_blocking_hook("always")
        ctx_pre = HookContext(event=HookEvent.PRE_TOOL_USE, run_id="x")
        ctx_post = HookContext(event=HookEvent.POST_TOOL_USE, run_id="x")
        pre = await hook(ctx_pre)
        post = await hook(ctx_post)
        assert pre.action is HookAction.BLOCK
        assert post.action is HookAction.ALLOW

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Runtime integration: hooks fire on the right events
# ---------------------------------------------------------------------------


async def test_stop_hook_fires_on_completion() -> None:
    sink: list[HookContext] = []
    hooks = HookRegistry()
    hooks.register(HookEvent.STOP, make_logging_hook(sink))
    runtime = AgentRuntime(
        provider=FakeProvider(responses=[ModelResponse(content="done")]),
        event_store=InMemoryEventStore(),
        policy=LoopPolicy(max_steps=4),
        hooks=hooks,
    )

    await runtime.run("task")

    assert len(sink) == 1
    assert sink[0].event is HookEvent.STOP
    assert sink[0].run_id


async def test_notification_hook_fires_on_consecutive_errors() -> None:
    sink: list[HookContext] = []
    hooks = HookRegistry()
    hooks.register(HookEvent.NOTIFICATION, make_logging_hook(sink))

    class _FlakyProvider(FakeProvider):
        async def generate(self, request):  # type: ignore[override]
            raise RuntimeError("provider boom")

    runtime = AgentRuntime(
        provider=_FlakyProvider(responses=[]),
        event_store=InMemoryEventStore(),
        policy=LoopPolicy(max_steps=5, consecutive_error_limit=2),
        hooks=hooks,
    )

    result = await runtime.run("task")
    assert result.stop_reason.value == "consecutive_errors"
    assert any(ctx.event is HookEvent.NOTIFICATION for ctx in sink)


async def test_pre_tool_hook_can_block_via_callback() -> None:
    """A PreToolUse hook returning BLOCK should deny without prompting."""

    asked: list[ToolCall] = []

    async def approval(call: ToolCall) -> bool:
        asked.append(call)
        return True

    hooks = HookRegistry()
    hooks.register(HookEvent.PRE_TOOL_USE, make_blocking_hook("no destructive shell"))
    runtime = AgentRuntime(
        provider=FakeProvider(
            responses=[
                ModelResponse(
                    tool_call=ToolCall(name="run_shell", arguments={"command": "rm -rf /"})
                )
            ]
        ),
        event_store=InMemoryEventStore(),
        policy=LoopPolicy(max_steps=3),
        approval_callback=approval,
        hooks=hooks,
    )

    result = await runtime.run("task")
    assert result.stop_reason.value == "policy_denied"
    # Approval was never asked because the hook vetoed the call.
    assert asked == []
