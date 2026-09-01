"""Tests for the ``web_search`` tool — DuckDuckGo HTML parsing + injection seam."""

from __future__ import annotations

from typing import Any

import pytest

from avo.app_tools.web_search import (
    DuckDuckGoClient,
    WebSearchArguments,
    WebSearchError,
    _parse_ddg_html,
    _search_async,
    decode_ddg_href,
    web_search_tool,
)


class _FakeClient:
    def __init__(self, hits: list[dict[str, str]] | Exception) -> None:
        self._hits = hits
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        self.calls.append({"query": query, "max_results": max_results})
        if isinstance(self._hits, Exception):
            raise self._hits
        return self._hits


_DDG_HTML = """
<html><body>
<a class="result__a" href="https://html.duckduckgo.com/html/?uddg=https%3A%2F%2Fexample.com%2Fa">Alpha</a>
<a class="result__snippet" class="x">First hit snippet.</a>

<a class="result__a" href="https://example.com/b">Beta</a>
<a class="result__snippet">Second hit.</a>

<a class="result__a" href="javascript:alert(1)">Bad</a>
<a class="result__snippet">Will be filtered.</a>
</body></html>
"""


def test_parse_ddg_html_unwraps_uddg() -> None:
    hits = _parse_ddg_html(_DDG_HTML, max_results=10)
    assert [h["title"] for h in hits] == ["Alpha", "Beta"]
    assert hits[0]["url"] == "https://example.com/a"
    assert hits[1]["url"] == "https://example.com/b"
    assert hits[0]["snippet"] == "First hit snippet."
    assert "Bad" not in [h["title"] for h in hits]


def test_parse_ddg_html_respects_max_results() -> None:
    hits = _parse_ddg_html(_DDG_HTML, max_results=1)
    assert len(hits) == 1
    assert hits[0]["title"] == "Alpha"


def test_parse_ddg_html_returns_empty_on_no_hits() -> None:
    assert _parse_ddg_html("<html></html>", max_results=10) == []


def test_decode_ddg_href_passthrough() -> None:
    assert decode_ddg_href("https://example.com/x") == "https://example.com/x"


async def test_search_async_dispatches_to_client() -> None:
    client = _FakeClient([{"title": "x", "url": "https://x/", "snippet": "y"}])
    result = await _search_async(client, WebSearchArguments(query="x"))
    assert result == [{"title": "x", "url": "https://x/", "snippet": "y"}]
    assert client.calls == [{"query": "x", "max_results": 10}]


async def test_search_async_propagates_client_errors() -> None:
    client = _FakeClient(WebSearchError("upstream down"))
    with pytest.raises(WebSearchError, match="upstream down"):
        await _search_async(client, WebSearchArguments(query="x"))


def test_parse_ddg_html_drops_bad_scheme_results() -> None:
    # Garbage-scheme URLs are filtered out silently rather than raised —
    # the parser is for normal search results and never trusts inbound
    # HTML to behave.
    hits = _parse_ddg_html(
        '<a class="result__a" href="file:///etc/passwd">x</a>\n<a class="result__snippet">y</a>',
        max_results=5,
    )
    assert hits == []


def test_parse_ddg_html_drops_missing_host_results() -> None:
    hits = _parse_ddg_html(
        '<a class="result__a" href="https://">x</a>\n<a class="result__snippet">y</a>',
        max_results=5,
    )
    assert hits == []


def test_search_arguments_reject_empty_query() -> None:
    with pytest.raises(ValueError):
        WebSearchArguments(query="")


def test_search_tool_metadata_describes_contract() -> None:
    tool = web_search_tool()
    metadata = tool.metadata
    assert metadata.name == "web_search"
    assert "query" in metadata.input_schema["properties"]


def test_duckduckgo_client_rejects_when_httpx_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import avo.app_tools.web_search as mod

    monkeypatch.setattr(mod, "_httpx", None)
    client = DuckDuckGoClient()
    import asyncio

    with pytest.raises(WebSearchError, match="requires httpx"):
        asyncio.run(client.search("x", max_results=5))
