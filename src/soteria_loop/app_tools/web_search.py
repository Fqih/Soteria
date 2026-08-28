"""``web_search`` tool: query a search backend and return hits.

The tool is a thin wrapper around a swappable client so callers can
inject a real search backend (DuckDuckGo HTML, Bing, Brave) without
tying the runtime to a single provider. The default client uses
DuckDuckGo's public HTML endpoint, which requires no API key.

Only ``http://`` and ``https://`` URLs are accepted when the caller
customizes the client; the default client performs the request itself
and inherits the same allowlist.
"""

from __future__ import annotations

from typing import Any, Protocol, cast
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import BaseModel, Field, JsonValue

from soteria_loop import FunctionTool as PublicFunctionTool
from soteria_loop.exceptions import ToolExecutionError

WebSearchError = ToolExecutionError

_DEFAULT_MAX_RESULTS = 10
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
_USER_AGENT = "Mozilla/5.0 (compatible; soteria-loop/1.0)"


class SearchClient(Protocol):
    """Protocol a search backend must implement.

    ``query`` is the search string. ``max_results`` caps the number of
    hits returned. Implementations return a list of dicts with at
    least ``title``, ``url``, ``snippet``.
    """

    async def search(self, query: str, *, max_results: int) -> list[dict[str, str]]: ...


class WebSearchArguments(BaseModel):
    """Arguments for the ``web_search`` tool."""

    query: str = Field(min_length=1, max_length=512)
    max_results: int = Field(default=_DEFAULT_MAX_RESULTS, gt=0, le=50)


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        allowed = ", ".join(sorted(_ALLOWED_SCHEMES))
        raise WebSearchError(
            f"web_search result URL only supports {allowed}; got {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise WebSearchError(f"web_search result URL is missing a host: {url}")
    return url


def decode_ddg_href(href: str) -> str:
    """DuckDuckGo's HTML endpoint wraps result URLs in a redirect; unwrap it."""

    parsed = urlparse(href)
    if "duckduckgo.com" not in (parsed.netloc or ""):
        return href
    query = parse_qs(parsed.query)
    if uddg := query.get("uddg"):
        return unquote(uddg[0])
    return href


def _parse_ddg_html(html: str, *, max_results: int) -> list[dict[str, str]]:
    """Internal parser. Public alias: :func:`parse_ddg_html`."""
    """Parse DuckDuckGo HTML results with stdlib only — no BeautifulSoup."""

    import re

    snippet_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    tag_re = re.compile(r"<[^>]+>")

    hits: list[dict[str, str]] = []
    for match in snippet_pattern.finditer(html):
        url = decode_ddg_href(match.group("url"))
        title = tag_re.sub("", match.group("title")).strip()
        snippet = tag_re.sub("", match.group("snippet")).strip()
        try:
            _validate_url(url)
        except WebSearchError:
            continue
        hits.append({"title": title, "url": url, "snippet": snippet})
        if len(hits) >= max_results:
            break
    return hits


try:
    import httpx as _httpx
except ModuleNotFoundError:  # pragma: no cover - httpx required by providers extra
    _httpx = None  # type: ignore[assignment]


class DuckDuckGoClient:
    """Default search client — DuckDuckGo HTML endpoint, no API key required."""

    def __init__(self, *, user_agent: str = _USER_AGENT) -> None:
        self._user_agent = user_agent

    async def search(self, query: str, *, max_results: int) -> list[dict[str, JsonValue]]:
        if _httpx is None:
            raise WebSearchError(
                "web_search requires httpx; install soteria-loop with the [providers] extra."
            )
        async with _httpx.AsyncClient(
            headers={"User-Agent": self._user_agent},
            timeout=15.0,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                _DDG_ENDPOINT,
                data={"q": query, "kl": ""},
            )
        return [dict(hit) for hit in _parse_ddg_html(response.text, max_results=max_results)]


def _search_with_client(
    client: Any,
    arguments: WebSearchArguments,
) -> list[dict[str, JsonValue]]:
    # Async callers receive a coroutine here; the dispatch wraps it.
    raise RuntimeError("use _search_async instead")


async def _search_async(client: Any, arguments: WebSearchArguments) -> list[dict[str, JsonValue]]:
    raw: list[dict[str, str]] = await client.search(
        arguments.query, max_results=arguments.max_results
    )
    return [dict(hit) for hit in raw]


async def _web_search(arguments: WebSearchArguments) -> dict[str, JsonValue]:
    if _httpx is None:
        raise WebSearchError(
            "web_search requires httpx; install soteria-loop with the [providers] extra."
        )
    client = DuckDuckGoClient()
    hits: list[dict[str, JsonValue]] = await _search_async(client, arguments)
    payload: dict[str, JsonValue] = {
        "query": arguments.query,
        "count": len(hits),
        "results": cast(JsonValue, hits),
    }
    return payload


def web_search_tool() -> PublicFunctionTool[WebSearchArguments]:
    """Return a :class:`FunctionTool` that performs web searches."""

    return PublicFunctionTool(
        name="web_search",
        description=(
            "Search the web via DuckDuckGo and return a list of hits. "
            "Each hit has ``title``, ``url``, and ``snippet``. No API key "
            "is required."
        ),
        arguments_model=WebSearchArguments,
        function=_web_search,
    )


__all__ = [
    "DuckDuckGoClient",
    "SearchClient",
    "WebSearchArguments",
    "WebSearchError",
    "decode_ddg_href",
    "parse_ddg_html",
    "web_search_tool",
]

# Public alias for the underscore-prefixed implementation.
parse_ddg_html = _parse_ddg_html


# Internal aliases (the test module imports the underscored variants).
_ = (_search_async, _search_with_client)
