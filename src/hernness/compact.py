"""Context-window compaction for long agent runs.

When the conversation history grows past the model's context window the
runtime must shrink it. ``compact_messages`` keeps a small head and tail
of the message list and either drops or summarises the middle. The
result is a shorter, still-coherent list that can replace the original
``context.messages`` in place.

The strategy is intentionally simple:

* Always preserve the first user message (``keep_first_user=True``).
  This anchors the run's original task in any future model context.
* Always preserve the last ``keep_last`` messages so the model sees the
  most recent decisions, tool calls, and observations.
* Either drop the middle silently or pass it through
  ``summarize_callable`` (an async provider call) and prepend a single
  ``system`` message carrying the summary.

The runtime itself never calls ``compact_messages`` automatically in
this version — operators can wire it into ``LoopPolicy`` or invoke it
manually via the ``/compact`` slash command in :mod:`hernness.chat`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import JsonValue

SummarizeCallable = Callable[[list[dict[str, JsonValue]]], Awaitable[str]]

_DEFAULT_KEEP_LAST = 8


def _message_text(message: dict[str, JsonValue]) -> str:
    """Return a printable summary of ``message`` for fallback compaction."""

    role = message.get("role")
    content = message.get("content")
    if isinstance(content, str):
        return f"[{role}] {content[:120]}"
    return f"[{role}] {str(content)[:120]}"


async def compact_messages(
    messages: list[dict[str, JsonValue]],
    *,
    keep_last: int = _DEFAULT_KEEP_LAST,
    keep_first_user: bool = True,
    summarize_callable: SummarizeCallable | None = None,
) -> list[dict[str, JsonValue]]:
    """Return a compacted copy of ``messages``.

    If ``summarize_callable`` is supplied the dropped middle is
    summarised into a single ``system`` message and prepended to the
    tail. Otherwise the middle is dropped silently.

    The function is a no-op when the input already fits in the
    ``keep_first_user + keep_last`` budget.
    """

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    if not messages:
        return list(messages)

    head: list[dict[str, JsonValue]] = []
    head_index = 0
    if keep_first_user:
        for index, message in enumerate(messages):
            if message.get("role") == "user":
                head = [message]
                head_index = index + 1
                break

    budget = len(head) + keep_last
    if len(messages) <= budget:
        return list(messages)

    middle = messages[head_index : len(messages) - keep_last]
    tail = list(messages[-keep_last:])

    if middle and summarize_callable is not None:
        summary = await summarize_callable(middle)
        summary_message: dict[str, JsonValue] = {
            "role": "system",
            "content": ("Summary of earlier conversation (compacted):\n" + (summary or "(empty)")),
        }
        return [*head, summary_message, *tail]

    if middle and summarize_callable is None:
        # No summarise callable: emit a placeholder system note so the
        # operator can still see that compaction happened.
        placeholder: dict[str, JsonValue] = {
            "role": "system",
            "content": (f"Compacted {len(middle)} earlier message(s); summarise disabled."),
        }
        return [*head, placeholder, *tail]

    return [*head, *tail]


def estimate_message_count(messages: list[dict[str, JsonValue]]) -> int:
    """Return the message count (cheap proxy for "how big is the context")."""

    return len(messages)


__all__ = ["SummarizeCallable", "compact_messages", "estimate_message_count"]


_ = _message_text  # referenced indirectly via summarise fallback; keep imported
