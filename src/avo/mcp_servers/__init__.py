"""Built-in MCP servers shipped with Avo.

Run any of them directly::

    python -m avo.mcp_servers.filesystem --root /path/to/workspace
    python -m avo.mcp_servers.sqlite --db /path/to/avo.db
    python -m avo.mcp_servers.git --cwd /path/to/repo
    python -m avo.mcp_servers.http_fetch

Each module exposes a :func:`build_server` factory so application code
can mount them via :class:`avo.mcp.MCPServer` instead of spawning a
subprocess (useful for in-process testing).
"""

from avo.mcp_servers.filesystem import build_server as filesystem_server
from avo.mcp_servers.git import build_server as git_server
from avo.mcp_servers.http_fetch import build_server as http_fetch_server
from avo.mcp_servers.sqlite import build_server as sqlite_server

__all__ = [
    "filesystem_server",
    "git_server",
    "http_fetch_server",
    "sqlite_server",
]
