"""Drive the built-in MCP filesystem server end-to-end.

This example shows the exact workflow users hit when they run::

    avo mcp add filesystem -- python -m avo.mcp_servers.filesystem
    avo chat            # REPL auto-attaches every registered server

We launch the server as a real stdio subprocess (the same argv
``avo mcp`` stores), perform the JSON-RPC handshake, list the
server's tools, and call ``read_file``. Then we wrap the MCP tools
in an Avo :class:`AgentRuntime` to prove MCP-wrapped tools behave
identically to first-party Avo tools.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from avo.mcp import MCPClient, MCPServer
from avo.providers import FakeProvider
from avo.runtime import AgentRuntime


def _spawn_server(root: Path) -> MCPServer:
    return MCPServer(
        command=[
            sys.executable,
            "-m",
            "avo.mcp_servers.filesystem",
            "--root",
            str(root),
        ],
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hello.txt").write_text("hello from mcp_demo.py\n", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "data.txt").write_text("nested file\n", encoding="utf-8")

        async with MCPClient(server=_spawn_server(root)) as client:
            tools = client.tools()
            tool_names = sorted(t.metadata.name for t in tools)
            print(f"Server info: {client.server.server_info}")
            print(f"Discovered tools: {tool_names}")

            read_file = next(t for t in tools if t.metadata.name == "read_file")
            list_dir = next(t for t in tools if t.metadata.name == "list_directory")
            search_files = next(t for t in tools if t.metadata.name == "search_files")

            read_result = await read_file.invoke({"path": "hello.txt"})
            print(f"read_file('hello.txt') → {read_result['text']!r}")

            list_result = await list_dir.invoke({"path": ".", "max_entries": 10})
            print(f"list_directory('.') → {list_result['text']!r}")

            search_result = await search_files.invoke({"pattern": "**/*.txt", "max_entries": 10})
            print(f"search_files('**/*.txt') → {search_result['text']!r}")

        # Wrap the same set of MCP tools in an AgentRuntime and load them
        # alongside a scripted FakeProvider. This mirrors what `avo chat`
        # does for every registered server on boot.
        async with MCPClient(server=_spawn_server(root)) as client:
            runtime = AgentRuntime(provider=FakeProvider([]), tools=client.tools())
            tools_summary = [t.name for t in runtime.tools.metadata]
            print(f"AgentRuntime loaded {len(runtime.tools.metadata)} MCP tool(s): {tools_summary}")


if __name__ == "__main__":
    asyncio.run(main())
