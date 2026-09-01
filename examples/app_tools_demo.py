"""Demonstration of the application tools package.

Run with::

    python examples/app_tools_demo.py

The example spins up a temporary workspace, registers the bundled
``read_file`` / ``write_file`` tools, wires a permissive approval
callback, and walks the runtime through a single-step "find the
README and add a line at the end" task via a ``FakeProvider``.

The demo is deterministic and offline — no API keys required.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from avo import AgentRuntime, FakeProvider, ModelResponse, ToolCall
from avo.app_tools.file_tools import bind_workspace, read_file_tool, write_file_tool
from avo.app_tools.workspace import Workspace


class WriteArgs(BaseModel):
    """Pydantic arguments model required by ``FunctionTool``."""

    path: str
    content: str


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = Workspace(root)

        # Pre-seed a small README inside the workspace.
        (root / "README.md.md").write_text(
            "Welcome to the demo workspace.\n",
            encoding="utf-8",
        )

        # Wire the read_file / write_file tools into a runtime.
        with bind_workspace(workspace):
            runtime = AgentRuntime(
                provider=FakeProvider(
                    [
                        ModelResponse(
                            tool_call=ToolCall(
                                tool_call_id="call-1",
                                name="write_file",
                                arguments={
                                    "path": "README.md.md",
                                    "content": "Welcome to the demo workspace.\nAppended line.\n",
                                },
                            )
                        ),
                        ModelResponse(content="done"),
                    ]
                ),
                tools=[read_file_tool(), write_file_tool()],
            )
            result = await runtime.run("Append a line to README.md.md")

        print(f"status: {result.status}")
        print(f"stop_reason: {result.stop_reason}")
        print(f"steps: {result.steps}")
        print(f"final README:\n{(root / 'README.md.md').read_text(encoding='utf-8')}")


if __name__ == "__main__":
    asyncio.run(main())
