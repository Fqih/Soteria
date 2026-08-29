"""Small, optional adapter for the local Lethe memory package.

Lethe remains an optional dependency. The adapter uses its stable
``MemoryStore.recall`` and ``MemoryStore.remember`` methods without importing
Lethe at module import time, so Hernness's core stays dependency-free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryProvider(Protocol):
    """Synchronous memory contract used to bound model context."""

    def recall(self, query: str, *, k: int = 5) -> list[object]:
        """Return relevant memory records."""

    def remember(
        self,
        content: str,
        *,
        session_id: str,
        tags: list[str] | None = None,
    ) -> object:
        """Persist one memory record."""


class LetheMemoryAdapter:
    """Adapt a Lethe ``MemoryStore`` to Hernness's context hooks.

    The adapter returns only bounded text snippets and never injects the full
    operational event log into the model context. Lethe itself must be
    installed separately (for example from ``../Lethe`` during development).
    """

    def __init__(self, store: MemoryProvider, *, recall_k: int = 5) -> None:
        if recall_k <= 0:
            raise ValueError("recall_k must be positive")
        self.store = store
        self.recall_k = recall_k

    def recall_text(self, query: str) -> list[str]:
        """Return memory content strings suitable for a context message."""

        return [
            str(getattr(item, "content", item))
            for item in self.store.recall(query, k=self.recall_k)
        ]

    def remember_output(self, content: str, *, session_id: str) -> None:
        """Persist a final agent output as a Lethe memory."""

        if content.strip():
            self.store.remember(content, session_id=session_id, tags=["soteria_output"])
