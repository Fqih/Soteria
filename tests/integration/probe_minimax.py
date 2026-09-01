"""Manual probe for the MiniMax 400 mystery.

Run with::

    AVO_MINIMAX_API_KEY=... python -m tests.integration.probe_minimax

What it does:

- prints the resolved endpoint for both API styles,
- prints the headers that would be sent (with the key redacted),
- prints the JSON body that would be sent,
- hits the real endpoint,
- prints the response status and body verbatim.

If MiniMax returns 400, the body usually identifies which field is
wrong. Paste the output back to the dev team.

This file is *not* a pytest test — invoke it directly. Keeping it
outside the test tree lets operators re-run it without a pytest
invocation, which is the right ergonomics for an interactive probe.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Make the project importable when run as ``python tests/integration/probe_minimax.py``.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from avo.providers.minimax import MiniMaxConfig, MiniMaxProvider  # noqa: E402

DEFAULT_BASE = "https://api.minimax.io"
DEFAULT_MODEL = "MiniMax-M3"


def _redact_key(value: str, key: str | None) -> str:
    """Return ``value`` with ``key`` replaced by ``[REDACTED]``."""

    if not key:
        return value
    return value.replace(key, "[REDACTED]")


async def _probe_one(style: str, api_key: str, model: str, base_url: str) -> None:
    print(f"\n=== style={style} ===")
    config = MiniMaxConfig.model_construct(model=model, base_url=base_url, api_style=style)  # type: ignore[arg-type]
    config._soteria_api_key = api_key

    print(f"endpoint  : {config.endpoint}")
    headers = config.headers()
    print(f"headers   : {_redact_key(repr(headers), api_key)}")

    provider = MiniMaxProvider(config, request_timeout_seconds=60.0)
    request_payload: dict[str, object] = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    }
    if style == "anthropic":
        request_payload["max_tokens"] = 64
    else:
        request_payload["max_tokens"] = 64
        request_payload["max_completion_tokens"] = 64
    print("payload   : " + json.dumps(request_payload, indent=2, sort_keys=True))

    try:
        response = await provider.generate(
            _build_minimal_request(),
        )
    except Exception as exc:
        print(f"\nERROR ({type(exc).__name__}):")
        print(_redact_key(str(exc), api_key))
        return
    finally:
        await provider.aclose()

    print("\nRESPONSE OK:")
    print(f"  content       : {response.content!r}")
    print(f"  tool_call     : {response.tool_call!r}")
    print(f"  usage         : {response.usage!r}")


def _build_minimal_request():
    from avo import ModelRequest

    return ModelRequest(
        run_id="probe-minimax",
        step=1,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        tools=[],
    )


async def main() -> int:
    api_key = os.environ.get("AVO_MINIMAX_API_KEY", "").strip()
    if not api_key:
        print("AVO_MINIMAX_API_KEY is not set; nothing to probe.", file=sys.stderr)
        return 2
    model = os.environ.get("AVO_MINIMAX_MODEL", "").strip() or DEFAULT_MODEL
    base_url = os.environ.get("AVO_MINIMAX_BASE_URL", "").strip() or DEFAULT_BASE

    await _probe_one("anthropic", api_key, model, base_url)
    await _probe_one("openai", api_key, model, base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
