"""HTTP fetch MCP server (stdlib urllib — no extra dependency).

One tool: ``fetch_url`` returns the body of a GET request as text with
status, content-type, and a size cap. Only ``http://`` and ``https://``
schemes are accepted; the response body is truncated at 1 MiB so a
misbehaving endpoint cannot fill the run context.

Run directly::

    python -m avo.mcp_servers.http_fetch
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from email.message import Message
from typing import cast

from pydantic import BaseModel, Field

from avo.mcp_servers._stdio import AvoMCPServer, user_error

MAX_FETCH_BYTES = 1_048_576  # 1 MiB


class _FetchArgs(BaseModel):
    url: str = Field(description="http:// or https:// URL to fetch.")
    max_bytes: int = Field(default=MAX_FETCH_BYTES, description="Hard cap on response bytes.")
    timeout: float = Field(default=15.0, description="Network timeout in seconds.")


def build_server() -> AvoMCPServer:
    server = AvoMCPServer(server_name="avo-http-fetch", server_version="0.1.0")

    @server.tool(
        name="fetch_url",
        description="Fetch a URL over HTTP(S) and return the body as text.",
        arguments_model=_FetchArgs,
    )
    async def _fetch(args: _FetchArgs) -> dict[str, object]:
        if not args.url.startswith(("http://", "https://")):
            user_error(f"unsupported URL scheme: {args.url}")
        request = urllib.request.Request(
            args.url,
            headers={"User-Agent": "avo-mcp-http-fetch/0.1", "Accept": "*/*"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            user_error(f"HTTP {exc.code}: {exc.reason}")
        except urllib.error.URLError as exc:
            user_error(f"network error: {exc.reason}")
        body = response.read(args.max_bytes + 1)
        truncated = len(body) > args.max_bytes
        body = body[: args.max_bytes]
        headers = cast(Message, response.headers)
        content_type = headers.get_content_type() if headers else "application/octet-stream"
        return {
            "url": args.url,
            "status": response.status,
            "content_type": content_type,
            "truncated": truncated,
            "body": body.decode("utf-8", errors="replace"),
        }

    return server


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="avo.mcp_servers.http_fetch")
    parser.parse_args(argv)  # no flags yet; reserved for future use
    return argparse.Namespace()


def main(argv: list[str] | None = None) -> int:
    import asyncio
    import sys as _sys

    _parse_args(_sys.argv[1:] if argv is None else argv)
    server = build_server()
    asyncio.run(server.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
