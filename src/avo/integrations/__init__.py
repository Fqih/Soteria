"""Optional integrations for Avo."""

from avo.integrations.lethe import LetheMemoryAdapter, MemoryProvider

__all__ = ["LetheMemoryAdapter", "MemoryProvider"]


def __getattr__(name: str) -> object:
    """Lazily expose LangChain bridge so the import is opt-in.

    Importing ``avo.integrations.langchain_bridge`` directly requires
    ``langchain-core`` to be installed; users who do not need the
    bridge should never pay that import cost.
    """

    if name in {"AvoLangChainModel", "avo_messages_from_lc", "avo_tool_from_lc"}:
        from avo.integrations import langchain_bridge

        return getattr(langchain_bridge, name)
    raise AttributeError(name)
