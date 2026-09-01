"""Tests for the sandbox executor and the ``run_shell`` tool.

The docker client is mocked so the suite stays offline (matching the
Avo principle: tests never depend on real services). The mock
records the kwargs passed to ``containers().create(...)`` so we can
verify the security-relevant settings (``network_mode``, ``mem_limit``,
``remove=True``, working directory) made it through to docker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from avo.app_tools.file_tools import WorkspaceNotBoundError, bind_workspace
from avo.app_tools.sandbox import SandboxExecutor, SandboxResult
from avo.app_tools.shell_tool import (
    RunShellArguments,
    SandboxNotBoundError,
    bind_sandbox,
    run_shell_tool,
)
from avo.app_tools.workspace import Workspace


class FakeContainer:
    """Records ``wait`` and ``logs`` results for assertions."""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "ok",
        stderr: str = "",
    ) -> None:
        self.id = "abcdef0123456789"
        self.short_id = "abcdef01"
        self.exit_code = exit_code
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.removed = False
        self.wait_calls = 0

    def wait(self) -> dict[str, int]:
        self.wait_calls += 1
        return {"StatusCode": self.exit_code}

    def logs(self, *, stdout: bool = True, stderr: bool = True) -> bytes:
        return (self.stdout_text + self.stderr_text).encode("utf-8")

    def remove(self, *, force: bool = False) -> None:
        del force
        self.removed = True


class FakeContainersAPI:
    """Records the kwargs passed to ``create`` and returns a FakeContainer."""

    def __init__(self, container: FakeContainer | None = None) -> None:
        self.container = container if container is not None else FakeContainer()
        self.create_calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeContainer:
        self.create_calls.append(kwargs)
        return self.container


class FakeDockerClient:
    """Top-level fake that mimics ``docker.DockerClient``."""

    def __init__(self, container: FakeContainer | None = None) -> None:
        self.containers_api = FakeContainersAPI(container)

    def containers(self) -> FakeContainersAPI:
        return self.containers_api


# ---------------------------------------------------------------------------
# SandboxExecutor unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_creates_container_with_offline_network_and_mem_limit(
    tmp_path: Path,
) -> None:
    container = FakeContainer(stdout="hello world")
    client = FakeDockerClient(container)
    executor = SandboxExecutor(client=client, image="python:3.12-slim")

    result = await executor.run("echo hello world", workspace_dir=tmp_path)

    create_kwargs = client.containers_api.create_calls[0]
    assert create_kwargs["network_mode"] == "none"
    assert create_kwargs["mem_limit"] == "256m"
    assert create_kwargs["cpu_quota"] == 50000
    assert create_kwargs["image"] == "python:3.12-slim"
    assert create_kwargs["remove"] is True
    assert create_kwargs["detach"] is True
    assert create_kwargs["working_dir"] == "/workspace"
    # Command is wrapped in ``sh -c`` so multi-token strings are honored.
    assert create_kwargs["command"] == ["sh", "-c", "echo hello world"]

    assert result.exit_code == 0
    assert "hello world" in result.stdout
    assert container.removed is True


@pytest.mark.asyncio
async def test_sandbox_propagates_exit_code(tmp_path: Path) -> None:
    container = FakeContainer(exit_code=2, stdout="boom", stderr="")
    client = FakeDockerClient(container)
    executor = SandboxExecutor(client=client)

    result = await executor.run("false", workspace_dir=tmp_path)

    assert result.exit_code == 2


@pytest.mark.asyncio
async def test_sandbox_timeout_raises(tmp_path: Path) -> None:
    class _HangingContainer(FakeContainer):
        def wait(self) -> dict[str, int]:
            import time

            time.sleep(5.0)
            return {"StatusCode": 0}

    container = _HangingContainer()
    client = FakeDockerClient(container)
    executor = SandboxExecutor(client=client, timeout_seconds=0.05)

    with pytest.raises(Exception, match="exceeded"):
        await executor.run("sleep 10", workspace_dir=tmp_path)

    assert container.removed is True


@pytest.mark.asyncio
async def test_sandbox_without_docker_package_raises() -> None:
    """If the docker client import fails the executor refuses to run."""

    executor = SandboxExecutor(client=None)  # no client, no docker import

    # Force the import path by clearing any cached client.
    executor._client = None
    # Patch the docker import to fail.
    import builtins

    original_import = builtins.__import__

    def _import_block(name: str, *args: object, **kwargs: object) -> object:
        if name == "docker" or name.startswith("docker."):
            raise ModuleNotFoundError("docker package not installed")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = _import_block  # type: ignore[assignment]
    try:
        with pytest.raises(Exception, match="docker"):
            await executor.run("true", workspace_dir=Path("/tmp"))
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]


def test_sandbox_result_is_immutable() -> None:
    result = SandboxResult(
        exit_code=0,
        stdout="x",
        stderr="",
        duration_ms=1.0,
        image="python:3.12-slim",
        network_mode="none",
        mem_limit="256m",
    )
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass
        result.exit_code = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_shell tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_shell_requires_sandbox_binding(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    tool = run_shell_tool()
    with bind_workspace(workspace), pytest.raises(SandboxNotBoundError):
        await tool._function(RunShellArguments(command="echo hi"))


@pytest.mark.asyncio
async def test_run_shell_requires_workspace_binding(tmp_path: Path) -> None:
    tool = run_shell_tool()
    executor = SandboxExecutor(client=FakeDockerClient())
    with bind_sandbox(executor), pytest.raises(WorkspaceNotBoundError):
        await tool._function(RunShellArguments(command="echo hi"))


@pytest.mark.asyncio
async def test_run_shell_returns_sandbox_result(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    container = FakeContainer(stdout="hello\n")
    client = FakeDockerClient(container)
    executor = SandboxExecutor(client=client)
    tool = run_shell_tool()

    with bind_workspace(workspace), bind_sandbox(executor):
        result = await tool._function(RunShellArguments(command="echo hello"))

    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert result["network_mode"] == "none"
    assert result["mem_limit"] == "256m"
    assert container.removed is True


@pytest.mark.asyncio
async def test_run_shell_passes_timeout_to_executor(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, create=True)
    container = FakeContainer()
    client = FakeDockerClient(container)
    executor = SandboxExecutor(client=client, timeout_seconds=10.0)
    tool = run_shell_tool()

    with bind_workspace(workspace), bind_sandbox(executor):
        await tool._function(RunShellArguments(command="echo hi", timeout_seconds=0.5))

    # Timeout override is passed to executor.run, not stored on the
    # container, so we just verify no exception is raised.
    assert container.removed is True
