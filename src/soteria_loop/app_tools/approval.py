"""Approval policy builder for application tools.

The runtime invokes ``approval_callback(call)`` while in the
``APPROVAL_PENDING`` state (see ``src/soteria_loop/runtime.py`` and the
state-machine reference). This module produces a callable that consults
the ``SOTERIA_TOOLS_REQUIRE_APPROVAL`` environment variable to decide
which tool names must wait for explicit operator approval.

The variable is a comma- or whitespace-separated list of tool names.
Tool names not present are auto-approved without invoking any
callback machinery.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Protocol

from soteria_loop.models import ToolCall


class ApprovalDecision(Protocol):
    """A user-provided callable that returns True to approve a tool call."""

    def __call__(self, call: ToolCall) -> bool | Awaitable[bool]: ...


def _parse_env_list(environ: os._Environ[str] | dict[str, str]) -> set[str]:
    """Parse ``SOTERIA_TOOLS_REQUIRE_APPROVAL`` into a normalized name set."""

    raw = environ.get("SOTERIA_TOOLS_REQUIRE_APPROVAL", "")
    if not raw:
        return set()
    tokens: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        tokens.extend(chunk.split())
    return {name.strip() for name in tokens if name.strip()}


def build_approval_callback(
    environ: os._Environ[str] | dict[str, str] | None = None,
    *,
    on_require: Callable[[ToolCall], None] | None = None,
) -> ApprovalDecision:
    """Build an ``approval_callback`` that consults ``SOTERIA_TOOLS_REQUIRE_APPROVAL``.

    Args:
        environ: Mapping to read from. Defaults to ``os.environ``.
        on_require: Optional callback invoked whenever a tool requires
            approval. The runtime-supplied approval callback may be a
            coroutine; the operator-supplied one returned here is always
            synchronous.

    Behavior:
        - Tool names listed in ``SOTERIA_TOOLS_REQUIRE_APPROVAL`` → return
          ``False`` so the runtime stops the run with ``POLICY_DENIED``.
          This matches the documented behavior of a synchronous denial —
          the operator is expected to gate the actual decision outside the
          runtime. (The runtime treats ``False`` as denial.)
        - Tool names not in the list → return ``True`` immediately.

    Notes:
        For 0.1 this returns ``False`` rather than raising so the
        runtime stops deterministically. Operators wanting to inject a
        real interactive prompt can wrap this callback with their own
        ``on_require`` to escalate.
    """

    env: os._Environ[str] | dict[str, str] = os.environ if environ is None else environ
    required = _parse_env_list(env)

    def _callback(call: ToolCall) -> bool:
        if call.name not in required:
            return True
        if on_require is not None:
            on_require(call)
        # 0.1 behavior: required tools are denied. Operators wanting
        # interactive approval must wrap this callback.
        return False

    return _callback


def required_tool_names(
    environ: os._Environ[str] | dict[str, str] | None = None,
) -> set[str]:
    """Return the parsed set of tool names that require approval."""

    env: os._Environ[str] | dict[str, str] = os.environ if environ is None else environ
    return _parse_env_list(env)


__all__ = [
    "ApprovalDecision",
    "build_approval_callback",
    "required_tool_names",
]
