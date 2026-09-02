"""Prompt caching hints for LLM providers.

Modern APIs let callers mark stable prefixes so the server can reuse
the KV cache between turns instead of recomputing it. The shape
differs per provider:

* Anthropic Messages — ``cache_control: {"type": "ephemeral"}`` on
  the last ``content_block`` of the breakpoint message.
* OpenAI Chat Completions — caching is automatic; the only opt-in
  is ``prompt_cache_key`` for cache partitioning.
* Groq / Cerebras / Ollama — no client-side hint.

The runtime doesn't need to know the per-provider details. Callers
set ``ModelRequest.cache = True`` (optionally ``cache_prefix_messages``
to pin how many leading messages are part of the stable prefix),
and each provider translates that into the appropriate hint.

The default breakpoint strategy is "cache everything up to the last
two messages" — the assumption is that the system prompt + tool
schema + early conversation are stable, while the most recent user
turn + pending tool result are not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheBreakpoints:
    """Indices of messages that mark cache boundaries.

    ``prefix`` is the count of leading messages to consider stable
    (cache breakpoint inserted after them). ``tail`` is the count of
    trailing messages to also cache (typically the final user turn).
    """

    prefix: int
    tail: int = 1

    @property
    def prefix_index(self) -> int:
        """Return the message index whose last block carries a cache_control marker."""

        return max(0, self.prefix - 1)


def stable_prefix_size(messages: Iterable[Mapping[str, Any]]) -> int:
    """Return the count of leading messages that look stable.

    A message is considered stable when its role is ``system`` or
    ``assistant`` (a finished turn). ``user`` and ``tool`` messages
    are treated as the moving frontier and end the prefix.
    """

    count = 0
    for message in messages:
        role = message.get("role")
        if role in ("system", "assistant"):
            count += 1
            continue
        break
    return count


def compute_breakpoints(
    messages: Sequence[Mapping[str, Any]],
    *,
    prefix_override: int | None = None,
    tail: int = 1,
) -> CacheBreakpoints:
    """Return the cache breakpoints to apply for this request.

    ``prefix_override`` lets the caller pin a custom prefix length;
    otherwise the heuristic in :func:`stable_prefix_size` is used.
    """

    total = len(messages)
    if total == 0:
        return CacheBreakpoints(prefix=0, tail=0)
    prefix = total - tail if prefix_override is None else min(prefix_override, total)
    return CacheBreakpoints(prefix=max(0, prefix), tail=min(tail, total))


def annotate_anthropic_messages(
    messages: list[dict[str, Any]],
    breakpoints: CacheBreakpoints,
) -> None:
    """Mark the breakpoint and tail messages with ``cache_control``.

    Mutates ``messages`` in place — providers build the payload once
    per request so in-place edits are fine and avoid copying the
    message list. Adds ``cache_control: {"type": "ephemeral"}`` to
    the final content block of each marker message.
    """

    indices = _marker_indices(messages, breakpoints)
    for index in indices:
        message = messages[index]
        content = message.get("content")
        if isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str):
            message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]


def cache_key_for_request(
    *,
    run_id: str,
    step: int,
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Derive a stable cache key for OpenAI-style cache partitioning.

    The key is deterministic per ``(run_id, step)`` and the SHA-256
    prefix of the JSON-encoded message tail. Identical prefixes share
    a key; divergent tails diverge it.
    """

    digest = hashlib.sha256()
    for message in messages:
        digest.update(repr(_serialise(message)).encode("utf-8"))
    return f"avo:{run_id}:{step}:{digest.hexdigest()[:16]}"


def _marker_indices(
    messages: Sequence[Mapping[str, Any]],
    breakpoints: CacheBreakpoints,
) -> list[int]:
    """Return the indices to mark with cache_control."""

    total = len(messages)
    indices: set[int] = set()
    if breakpoints.prefix > 0 and total > 0:
        indices.add(min(breakpoints.prefix - 1, total - 1))
    if breakpoints.tail > 0 and total > 0:
        indices.add(total - 1)
    return sorted(indices)


def _serialise(value: Any) -> Any:
    """Stable JSON-like serialisation: dict keys sorted recursively."""

    if isinstance(value, dict):
        return {k: _serialise(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_serialise(item) for item in value]
    return value


__all__ = [
    "CacheBreakpoints",
    "annotate_anthropic_messages",
    "cache_key_for_request",
    "compute_breakpoints",
    "stable_prefix_size",
]
