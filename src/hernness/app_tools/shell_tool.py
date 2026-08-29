"""``run_shell`` tool: a single shell command inside a docker sandbox.

The tool binds a :class:`SandboxExecutor` to the active workspace via
the same context-manager pattern used by ``file_tools``. The executor
is shared across calls so the same client (and same image / network /
memory settings) is reused; each invocation still gets a fresh
container.

``run_shell`` never calls :mod:`subprocess` on the host. The sandbox
executor is the only path to the shell, and the executor's default
``network_mode="none"`` keeps the container offline.

The workspace binding is read from :func:`file_tools._current_workspace`
so ``run_shell`` participates in the same workspace check as
``read_file`` and ``write_file``. Tools that bind a workspace do not
need to bind the sandbox separately — but they may if they want a
custom executor.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from pydantic import BaseModel, Field, JsonValue

from hernness import FunctionTool as PublicFunctionTool

from .file_tools import WorkspaceNotBoundError, _current_workspace
from .sandbox import SandboxExecutor, SandboxResult


class RunShellArguments(BaseModel):
    """Arguments for the ``run_shell`` tool."""

    command: str = Field(min_length=1, description="Shell command to run inside the sandbox")
    timeout_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Per-call timeout; falls back to executor default",
    )


class SandboxNotBoundError(RuntimeError):
    """Raised when ``run_shell`` is invoked without an active sandbox binding."""


_executor_stack: list[SandboxExecutor] = []


@contextmanager
def bind_sandbox(executor: SandboxExecutor) -> Generator[None, None, None]:
    """Push ``executor`` onto the active binding for the duration of the block."""

    _executor_stack.append(executor)
    try:
        yield
    finally:
        _executor_stack.pop()


def _current_executor() -> SandboxExecutor:
    if not _executor_stack:
        raise SandboxNotBoundError(
            "run_shell invoked without an active sandbox; wrap the run in "
            "hernness.app_tools.shell_tool.bind_sandbox(...)"
        )
    return _executor_stack[-1]


async def _run_shell(arguments: RunShellArguments) -> JsonValue:
    executor = _current_executor()
    # Read the workspace from the file_tools binding so the sandbox
    # shares the same containment boundary as read_file / write_file.
    workspace_root = _current_workspace().root
    result: SandboxResult = await executor.run(
        arguments.command,
        workspace_dir=workspace_root,
        timeout_seconds=arguments.timeout_seconds,
    )
    payload: dict[str, JsonValue] = {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
        "image": result.image,
        "network_mode": result.network_mode,
        "mem_limit": result.mem_limit,
    }
    return payload


def run_shell_tool() -> PublicFunctionTool[RunShellArguments]:
    """Return a :class:`FunctionTool` that runs a shell command inside a sandbox."""

    return PublicFunctionTool(
        name="run_shell",
        description=(
            "Run a single shell command inside an ephemeral, network-isolated "
            "docker container. The container has a memory and CPU cap and is "
            "removed as soon as the command exits. The command runs in the "
            "active workspace."
        ),
        arguments_model=RunShellArguments,
        function=_run_shell,
    )


__all__ = [
    "RunShellArguments",
    "SandboxNotBoundError",
    "WorkspaceNotBoundError",
    "bind_sandbox",
    "run_shell_tool",
]
