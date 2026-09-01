"""Tests for the context-window compaction helper."""

from __future__ import annotations

import pytest

from avo.compact import compact_messages, estimate_message_count


def _messages(n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i in range(n):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m-{i}"})
    return out  # type: ignore[return-value]


async def test_compact_is_noop_when_within_budget() -> None:
    msgs = _messages(4)
    result = await compact_messages(msgs, keep_last=4, keep_first_user=True)
    assert result == msgs


async def test_compact_drops_middle_when_no_summariser() -> None:
    msgs = _messages(20)
    result = await compact_messages(msgs, keep_last=4, keep_first_user=True)
    # head (first user) + 1 placeholder system + 4 tail = 6 messages
    assert len(result) == 6
    assert result[0]["role"] == "user"
    assert "Compacted" in str(result[1]["content"])
    assert result[-1] == msgs[-1]


async def test_compact_summarises_middle_when_callable_provided() -> None:
    msgs = _messages(20)

    async def summariser(middle: list[dict[str, object]]) -> str:
        return " | ".join(str(m.get("content")) for m in middle)

    result = await compact_messages(
        msgs,
        keep_last=4,
        keep_first_user=True,
        summarize_callable=summariser,
    )
    assert len(result) == 6
    assert result[0]["role"] == "user"
    summary_msg = result[1]
    assert summary_msg["role"] == "system"
    assert "Summary of earlier conversation" in str(summary_msg["content"])
    assert "m-1" in str(summary_msg["content"])
    assert "m-15" in str(summary_msg["content"])
    assert result[-1] == msgs[-1]


async def test_compact_preserves_first_user_when_present() -> None:
    msgs = [
        {"role": "system", "content": "you are"},
        {"role": "user", "content": "anchor"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "real task"},
    ]
    # Add filler to force compaction
    filler = [{"role": "user", "content": f"f-{i}"} for i in range(10)]
    msgs.extend(filler)
    result = await compact_messages(
        msgs,
        keep_last=3,
        keep_first_user=True,
    )
    # Should not pick the system message as head.
    assert result[0] == {"role": "user", "content": "anchor"}


async def test_compact_omits_first_user_when_disabled() -> None:
    msgs = _messages(20)
    result = await compact_messages(msgs, keep_last=4, keep_first_user=False)
    # No head; placeholder + 4 tail
    assert len(result) == 5
    assert result[0]["role"] == "system"


async def test_compact_handles_empty_input() -> None:
    result = await compact_messages([])
    assert result == []


async def test_compact_rejects_non_positive_keep_last() -> None:
    with pytest.raises(ValueError, match="keep_last"):
        await compact_messages(_messages(3), keep_last=0)


async def test_compact_summarise_callable_receives_middle_only() -> None:
    msgs = _messages(20)
    seen: list[int] = []

    async def summariser(middle: list[dict[str, object]]) -> str:
        seen.append(len(middle))
        return "summary"

    await compact_messages(
        msgs,
        keep_last=4,
        keep_first_user=True,
        summarize_callable=summariser,
    )
    # 20 messages - 1 head - 4 tail = 15 middle
    assert seen == [15]


async def test_estimate_message_count_returns_len() -> None:
    assert estimate_message_count([]) == 0
    assert estimate_message_count(_messages(7)) == 7


async def test_compact_returns_new_list_not_in_place() -> None:
    msgs = _messages(20)
    original_len = len(msgs)
    result = await compact_messages(msgs, keep_last=4)
    assert len(msgs) == original_len
    assert result is not msgs
