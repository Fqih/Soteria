"""Tests for :mod:`avo.providers.prompt_cache` and provider integration."""

from __future__ import annotations

from typing import Any

from avo.providers.anthropic import AnthropicConfig, AnthropicProvider
from avo.providers.prompt_cache import (
    CacheBreakpoints,
    annotate_anthropic_messages,
    cache_key_for_request,
    compute_breakpoints,
    stable_prefix_size,
)


def _messages(*specs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": text} for role, text in specs]


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def test_stable_prefix_size_counts_system_and_assistant_only() -> None:
    messages = _messages(
        ("system", "you are a helper"),
        ("assistant", "ready"),
        ("user", "first question"),
        ("assistant", "answer"),
        ("user", "follow up"),
    )
    assert stable_prefix_size(messages) == 2


def test_stable_prefix_size_empty_when_starts_with_user() -> None:
    assert stable_prefix_size(_messages(("user", "hi"))) == 0


def test_compute_breakpoints_defaults() -> None:
    messages = _messages(
        ("system", "s"),
        ("assistant", "a1"),
        ("user", "u1"),
        ("assistant", "a2"),
        ("user", "u2"),
    )
    breakpoints = compute_breakpoints(messages)
    # 5 messages, default tail=1 → prefix=4
    assert breakpoints.prefix == 4
    assert breakpoints.tail == 1


def test_compute_breakpoints_with_override() -> None:
    messages = _messages(
        ("system", "s"),
        ("assistant", "a1"),
        ("user", "u1"),
        ("assistant", "a2"),
    )
    breakpoints = compute_breakpoints(messages, prefix_override=2)
    assert breakpoints.prefix == 2


def test_compute_breakpoints_empty_messages() -> None:
    breakpoints = compute_breakpoints([])
    assert breakpoints == CacheBreakpoints(prefix=0, tail=0)


# ---------------------------------------------------------------------------
# Anthropic annotation
# ---------------------------------------------------------------------------


def test_annotate_anthropic_messages_marks_breakpoint_and_tail() -> None:
    messages = _messages(
        ("system", "s"),
        ("assistant", "a1"),
        ("user", "u1"),
        ("assistant", "a2"),
        ("user", "u2"),
    )
    breakpoints = compute_breakpoints(messages)
    annotate_anthropic_messages(messages, breakpoints)
    # Indices [3, 4] are marked: 3 = prefix boundary ("a2"), 4 = tail ("u2").
    assert "cache_control" not in messages[0]
    assert "cache_control" not in messages[1]
    assert "cache_control" not in messages[2]
    assert messages[3] == {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "a2",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    assert messages[4] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "u2",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


def test_annotate_anthropic_messages_preserves_list_content() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
                {"type": "text", "text": "more"},
            ],
        }
    ]
    annotate_anthropic_messages(messages, CacheBreakpoints(prefix=1, tail=1))
    last_block = messages[0]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}
    # First block must not be mutated
    assert "cache_control" not in messages[0]["content"][0]


# ---------------------------------------------------------------------------
# Cache key derivation
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic_for_same_messages() -> None:
    messages = _messages(("user", "hi"), ("assistant", "hello"))
    key_a = cache_key_for_request(run_id="r1", step=1, messages=messages)
    key_b = cache_key_for_request(run_id="r1", step=1, messages=messages)
    assert key_a == key_b


def test_cache_key_changes_with_step() -> None:
    messages = _messages(("user", "hi"))
    assert cache_key_for_request(run_id="r1", step=1, messages=messages) != cache_key_for_request(
        run_id="r1", step=2, messages=messages
    )


def test_cache_key_changes_with_messages() -> None:
    assert cache_key_for_request(
        run_id="r1", step=1, messages=_messages(("user", "a"))
    ) != cache_key_for_request(run_id="r1", step=1, messages=_messages(("user", "b")))


# ---------------------------------------------------------------------------
# Anthropic provider integration
# ---------------------------------------------------------------------------


def test_anthropic_provider_payload_omits_cache_by_default() -> None:
    config = AnthropicConfig(model="claude-test")
    config._api_key = "sk-ant"
    provider = AnthropicProvider(config)
    request_payload = provider._build_payload(
        _request(
            messages=_messages(("system", "s"), ("user", "hi")),
            cache=False,
        )
    )
    for message in request_payload["messages"]:
        if isinstance(message["content"], list):
            for block in message["content"]:
                assert "cache_control" not in block
        elif isinstance(message["content"], dict):  # tool_use
            assert "cache_control" not in message["content"]


def test_anthropic_provider_payload_adds_cache_control_when_enabled() -> None:
    config = AnthropicConfig(model="claude-test")
    config._api_key = "sk-ant"
    provider = AnthropicProvider(config)
    payload = provider._build_payload(
        _request(
            messages=_messages(
                ("system", "s"),
                ("assistant", "a1"),
                ("user", "u1"),
                ("assistant", "a2"),
            ),
            cache=True,
            cache_prefix_messages=3,
        )
    )
    # After stripping the system message, anthropic_messages has 3 entries
    # [a1, u1, a2]. With prefix_override=3, the breakpoint lands on
    # ``a2`` (index 2) and the tail marker also lands on ``a2``.
    breakpoint_message = payload["messages"][2]
    assert breakpoint_message["content"] == [
        {
            "type": "text",
            "text": "a2",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _request(
    *, messages: list[dict[str, Any]], cache: bool, cache_prefix_messages: int | None = None
):
    from avo import ModelRequest

    return ModelRequest(
        run_id="run-1",
        step=1,
        messages=messages,  # type: ignore[arg-type]
        cache=cache,
        cache_prefix_messages=cache_prefix_messages,
    )
