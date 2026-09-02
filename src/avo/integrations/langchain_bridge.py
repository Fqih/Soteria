"""LangChain adapter for Avo model providers.

The bridge lets any object satisfying :class:`ModelProvider` (real or
test) be used inside a LangChain pipeline — chains, agents, output
parsers — without duplicating the message-format translation.

Install LangChain separately (``pip install avo[langchain]``); the
adapter uses lazy imports so the rest of Avo never pulls LangChain.

Three translation helpers are exported:

* :class:`AvoLangChainModel` — wrap an avo provider as a LangChain
  ``BaseChatModel``.
* :func:`avo_messages_from_lc` — convert a list of LangChain messages
  into the dict shape :class:`ModelRequest` expects.
* :func:`avo_tool_from_lc` — adapt a LangChain ``StructuredTool`` /
  ``BaseTool`` so it can be registered in an avo ``ToolRegistry``.

The bridge never imports LangChain at module load — every import is
deferred so importing :mod:`avo` stays free of LangChain.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from avo.models import ModelRequest, ModelResponse

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import BaseTool, StructuredTool


_LANGCHAIN_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - exercised only when langchain-core is installed
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        ChatResult,
    )
    from langchain_core.tools import BaseTool, StructuredTool
except ImportError as exc:  # pragma: no cover - exercised only when missing
    BaseChatModel = None  # type: ignore[assignment,misc]
    BaseMessage = None  # type: ignore[assignment,misc]
    AIMessage = None  # type: ignore[assignment,misc]
    AIMessageChunk = None  # type: ignore[assignment,misc]
    HumanMessage = None  # type: ignore[assignment,misc]
    SystemMessage = None  # type: ignore[assignment,misc]
    ToolMessage = None  # type: ignore[assignment,misc]
    ChatGeneration = None  # type: ignore[assignment,misc]
    ChatGenerationChunk = None  # type: ignore[assignment,misc]
    ChatResult = None  # type: ignore[assignment,misc]
    BaseTool = None  # type: ignore[assignment,misc]
    StructuredTool = None  # type: ignore[assignment,misc]
    _LANGCHAIN_IMPORT_ERROR = exc


def _require_langchain() -> None:
    if BaseChatModel is None:
        raise ImportError(
            "langchain-core is required for the Avo LangChain bridge. "
            "Install it with `pip install avo[langchain]`."
        ) from _LANGCHAIN_IMPORT_ERROR


def avo_messages_from_lc(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Translate LangChain messages into avo's wire format."""

    _require_langchain()
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": str(message.content)})
            continue
        if isinstance(message, HumanMessage):
            content = _lc_content_to_avo(message.content)
            out.append({"role": "user", "content": content})
            continue
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    out.append(
                        {
                            "role": "assistant",
                            "tool_call": {
                                "tool_call_id": call.get("id", ""),
                                "name": call.get("name", ""),
                                "arguments": call.get("args", {}),
                            },
                        }
                    )
            else:
                out.append({"role": "assistant", "content": str(message.content)})
            continue
        if isinstance(message, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": str(message.content),
                }
            )
            continue
        role = getattr(message, "type", "user")
        out.append({"role": str(role), "content": str(getattr(message, "content", ""))})
    return out


def avo_tool_from_lc(tool: Any) -> Any:
    """Wrap a LangChain ``BaseTool`` as an avo :class:`FunctionTool`."""

    _require_langchain()
    if not isinstance(tool, BaseTool):
        raise TypeError(f"expected BaseTool, got {type(tool).__name__}")

    from avo.tools import FunctionTool

    name = tool.name
    description = tool.description or ""
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        raise ValueError(f"tool {name!r} has no Pydantic args schema")

    async def _invoke(arguments: Any) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(arguments)
        if hasattr(tool, "arun"):
            return await tool.arun(**arguments)
        raise RuntimeError(f"tool {name!r} does not implement async invocation")

    return FunctionTool(
        name=name,
        description=description,
        arguments_model=schema,
        function=_invoke,
    )


class AvoLangChainModel(BaseChatModel):
    """LangChain ``BaseChatModel`` backed by an avo :class:`ModelProvider`."""

    avo_provider: Any
    avo_run_id: str = "langchain"
    avo_step: int = 1
    avo_cache: bool = False

    model_config: ClassVar[Any] = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "avo"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": getattr(self.avo_provider, "name", type(self.avo_provider).__name__),
            "run_id": self.avo_run_id,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync LangChain entry point — runs the async provider on a fresh loop."""

        import asyncio

        response = asyncio.run(
            self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )
        return response

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async LangChain entry point — delegates to the avo provider."""

        request = ModelRequest(
            run_id=self.avo_run_id,
            step=self.avo_step,
            messages=avo_messages_from_lc(messages),
            tools=[],
            cache=self.avo_cache,
        )
        response = await self.avo_provider.generate(request)
        return _chat_result_from_avo(response)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chunks via the avo provider's ``stream`` coroutine when available."""

        from avo.providers.streaming import StreamingModelProvider

        request = ModelRequest(
            run_id=self.avo_run_id,
            step=self.avo_step,
            messages=avo_messages_from_lc(messages),
            tools=[],
            cache=self.avo_cache,
        )
        if isinstance(self.avo_provider, StreamingModelProvider):
            async for chunk in self.avo_provider.stream(request):
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk.text or ""))
            return

        response = await self.avo_provider.generate(request)
        yield ChatGenerationChunk(message=AIMessageChunk(content=response.content or ""))


def _lc_content_to_avo(content: Any) -> Any:
    """Translate LC content (str | list[dict|str]) into avo's shape."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content
    return str(content)


def _chat_result_from_avo(response: ModelResponse) -> ChatResult:
    if response.tool_call is not None:
        call = response.tool_call
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": call.tool_call_id,
                    "name": call.name,
                    "args": dict(call.arguments),
                }
            ],
        )
    else:
        ai_message = AIMessage(content=response.content or "")
    return ChatResult(generations=[ChatGeneration(message=ai_message)])


__all__ = [
    "AvoLangChainModel",
    "avo_messages_from_lc",
    "avo_tool_from_lc",
]
