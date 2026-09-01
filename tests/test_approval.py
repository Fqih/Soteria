"""Tests for the ``AVO_TOOLS_REQUIRE_APPROVAL`` callback."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from avo.app_tools.approval import (
    build_approval_callback,
    required_tool_names,
)
from avo.models import ToolCall


def _call(name: str, tool_call_id: str = "id-1") -> ToolCall:
    return ToolCall(tool_call_id=tool_call_id, name=name, arguments={})


def test_empty_env_auto_approves() -> None:
    callback = build_approval_callback({})
    assert callback(_call("read_file")) is True
    assert callback(_call("run_shell")) is True


def test_env_with_one_tool_requires_approval() -> None:
    callback = build_approval_callback({"AVO_TOOLS_REQUIRE_APPROVAL": "run_shell"})
    assert callback(_call("run_shell")) is False
    assert callback(_call("read_file")) is True


def test_env_with_multiple_tools() -> None:
    callback = build_approval_callback({"AVO_TOOLS_REQUIRE_APPROVAL": "run_shell,write_file"})
    assert callback(_call("run_shell")) is False
    assert callback(_call("write_file")) is False
    assert callback(_call("read_file")) is True


def test_env_with_whitespace_and_newlines() -> None:
    callback = build_approval_callback(
        {"AVO_TOOLS_REQUIRE_APPROVAL": "run_shell\n write_file  read_file "}
    )
    assert callback(_call("run_shell")) is False
    assert callback(_call("write_file")) is False
    assert callback(_call("read_file")) is False


def test_empty_token_ignored() -> None:
    callback = build_approval_callback({"AVO_TOOLS_REQUIRE_APPROVAL": ",,,"})
    assert callback(_call("run_shell")) is True


def test_on_require_called_for_required_tool() -> None:
    seen: list[str] = []

    def on_require(call: ToolCall) -> None:
        seen.append(call.name)

    callback = build_approval_callback(
        {"AVO_TOOLS_REQUIRE_APPROVAL": "run_shell"}, on_require=on_require
    )
    callback(_call("run_shell"))
    callback(_call("read_file"))
    assert seen == ["run_shell"]


def test_required_tool_names_helper() -> None:
    assert required_tool_names({}) == set()
    assert required_tool_names({"AVO_TOOLS_REQUIRE_APPROVAL": "a,b,c"}) == {"a", "b", "c"}
    assert required_tool_names({"AVO_TOOLS_REQUIRE_APPROVAL": " a , b ,c "}) == {
        "a",
        "b",
        "c",
    }


def test_callback_signature_returns_bool() -> None:
    callback = build_approval_callback({"AVO_TOOLS_REQUIRE_APPROVAL": "run_shell"})
    result = callback(_call("run_shell"))
    assert isinstance(result, bool)
    # Auto-approve branch returns True.
    auto = callback(_call("read_file"))
    assert auto is True


def test_callback_is_sync_only() -> None:
    """The factory must return a sync callable for the 0.1 contract."""

    import inspect

    callback = build_approval_callback({})
    assert not inspect.iscoroutinefunction(callback)
    assert inspect.isfunction(callback) or callable(callback)
    # Cast for type checkers; the call signature is sync.
    _ = cast("Callable[[ToolCall], bool]", callback)
