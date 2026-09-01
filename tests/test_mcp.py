"""Tests for the MCP client adapter.

The integration tests launch a tiny in-process Python MCP server (a
script that frames JSON-RPC over stdio) so the suite stays offline.
The unit tests cover framing and adapter logic with a fake server.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from avo.mcp import MCPClient, MCPError, MCPServer, _read_header, mcp_tool

_FAKE_SERVER_PATH = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


@pytest.fixture
def fake_server_script() -> Path:
    return _FAKE_SERVER_PATH


async def _with_client(script: Path) -> MCPClient:
    server = MCPServer(command=[sys.executable, str(script)])
    client = MCPClient(server=server)
    await client.__aenter__()
    return client


async def test_mcp_client_performs_handshake(fake_server_script: Path) -> None:
    client = await _with_client(fake_server_script)
    try:
        assert client.server.server_info == {"name": "fake", "version": "0.0.1"}
        tool_names = {tool.metadata.name for tool in client.tools()}
        assert tool_names == {"echo"}
    finally:
        await client.__aexit__(None, None, None)


async def test_mcp_client_calls_tool(fake_server_script: Path) -> None:
    client = await _with_client(fake_server_script)
    try:
        echo = next(t for t in client.tools() if t.metadata.name == "echo")
        result = await echo._function(echo._arguments_model(x="hello"))  # type: ignore[no-any-return]
        assert result["name"] == "echo"
        assert result["is_error"] is False
        assert "hello" in result["text"]
    finally:
        await client.__aexit__(None, None, None)


async def test_mcp_client_outside_context_raises() -> None:
    server = MCPServer(command=[sys.executable, "-c", "pass"])
    client = MCPClient(server=server)
    with pytest.raises(MCPError, match="outside"):
        client.tools()


async def test_mcp_server_requires_command() -> None:
    with pytest.raises(MCPError, match="non-empty command"):
        MCPServer(command=[])


async def test_read_header_parses_content_length() -> None:
    sent = b"Content-Length: 42\r\n\r\n"
    stream = asyncio.StreamReader()
    stream.feed_data(sent)
    stream.feed_eof()
    assert await _read_header(stream) == 42


def test_mcp_tool_requires_name() -> None:
    server = MCPServer(command=["python", "-c", "pass"])
    with pytest.raises(MCPError, match="missing name"):
        mcp_tool(server, {"description": "x"})
