"""Tests for the LangChain bridge.

The bridge is an optional integration. Tests skip cleanly when
``langchain-core`` is not installed, mirroring the offline-by-default
philosophy of the rest of the suite.
"""

from __future__ import annotations

from typing import Any

import pytest

langchain_core = pytest.importorskip("langchain_core")
pytest.importorskip("langchain_core.messages")
pytest.importorskip("langchain_core.language_models.chat_models")
pytest.importorskip("langchain_core.outputs")
pytest.importorskip("langchain_core.tools")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from avo import ModelRequest, ModelResponse, ToolCall  # noqa: E402
from avo.integrations.langchain_bridge import (  # noqa: E402
    AvoLangChainModel,
    avo_messages_from_lc,
    avo_tool_from_lc,
)


class _FakeProvider:
    """Minimal avo ``ModelProvider`` stub for the bridge."""

    def __init__(self, response: ModelResponse) -> None:
        self._response = response
        self.calls: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return self._response


# ---------------------------------------------------------------------------
# avo_messages_from_lc
# ---------------------------------------------------------------------------


def test_avo_messages_from_lc_translates_basic_roles() -> None:
    out = avo_messages_from_lc(
        [
            SystemMessage(content="be helpful"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
    )
    assert out == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_avo_messages_from_lc_emits_tool_calls_per_call() -> None:
    ai = AIMessage(
        content="",
        tool_calls=[
            {"id": "c1", "name": "read", "args": {"path": "a.txt"}},
            {"id": "c2", "name": "write", "args": {"path": "b.txt", "data": "x"}},
        ],
    )
    out = avo_messages_from_lc([ai])
    assert len(out) == 2
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_call"]["tool_call_id"] == "c1"
    assert out[1]["tool_call"]["name"] == "write"


def test_avo_messages_from_lc_passes_through_tool_result() -> None:
    out = avo_messages_from_lc([ToolMessage(content="ok", tool_call_id="c1")])
    assert out == [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]


def test_avo_messages_from_lc_preserves_list_content() -> None:
    msg = HumanMessage(content=[{"type": "text", "text": "describe this"}])
    out = avo_messages_from_lc([msg])
    assert out[0]["role"] == "user"
    assert isinstance(out[0]["content"], list)


# ---------------------------------------------------------------------------
# AvoLangChainModel
# ---------------------------------------------------------------------------


def test_langchain_model_generate_returns_ai_message() -> None:
    provider = _FakeProvider(ModelResponse(content="hi back"))
    model = AvoLangChainModel(avo_provider=provider, avo_run_id="r1", avo_step=1)
    result = model.invoke([HumanMessage(content="hi")])
    assert isinstance(result, AIMessage)
    assert result.content == "hi back"


def test_langchain_model_generates_tool_call() -> None:
    provider = _FakeProvider(
        ModelResponse(
            content=None,
            tool_call=ToolCall(tool_call_id="c1", name="read", arguments={"path": "a.txt"}),
        )
    )
    model = AvoLangChainModel(avo_provider=provider)
    result = model.invoke([HumanMessage(content="open a.txt")])
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "read"
    assert result.tool_calls[0]["args"] == {"path": "a.txt"}


def test_langchain_model_passes_messages_to_avo_provider() -> None:
    provider = _FakeProvider(ModelResponse(content="ok"))
    model = AvoLangChainModel(avo_provider=provider, avo_run_id="abc", avo_step=3)
    model.invoke(
        [
            SystemMessage(content="be terse"),
            HumanMessage(content="hi"),
        ]
    )
    sent = provider.calls[0]
    assert sent.run_id == "abc"
    assert sent.step == 3
    assert sent.messages[0]["role"] == "system"
    assert sent.messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# avo_tool_from_lc
# ---------------------------------------------------------------------------


class _EchoInput(BaseModel):
    text: str


def test_avo_tool_from_lc_adapts_structured_tool() -> None:
    def echo(text: str) -> str:
        return text

    lc_tool = StructuredTool.from_function(
        func=echo,
        name="echo",
        description="echo input",
        args_schema=_EchoInput,
    )
    avo_tool = avo_tool_from_lc(lc_tool)
    assert avo_tool.metadata.name == "echo"
    assert avo_tool.metadata.description == "echo input"
    assert avo_tool.metadata.input_schema == _EchoInput.model_json_schema()


def test_avo_tool_from_lc_rejects_non_tool() -> None:
    with pytest.raises(TypeError):
        avo_tool_from_lc("not a tool")


def test_avo_tool_from_lc_invokes_async_underlying() -> None:
    seen: list[Any] = []

    async def aecho(text: str) -> str:
        seen.append(text)
        return f"echo:{text}"

    lc_tool = StructuredTool.from_function(
        func=lambda text: None,
        coroutine=aecho,
        name="aecho",
        description="async echo",
        args_schema=_EchoInput,
    )
    avo_tool = avo_tool_from_lc(lc_tool)
    import asyncio

    value = asyncio.run(avo_tool.invoke({"text": "hi"}))
    assert value == "echo:hi"
    assert seen == ["hi"]
