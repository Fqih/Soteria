"""Optional Lethe context-window integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from avo.integrations import LetheMemoryAdapter


@dataclass
class Item:
    content: str


class MemoryStub:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, list[str]]] = []

    def recall(self, query: str, *, k: int = 5) -> list[Item]:
        assert query == "current task"
        assert k == 2
        return [Item("bounded memory")]

    def remember(self, content: str, *, session_id: str, tags: list[str] | None = None) -> None:
        self.saved.append((content, session_id, list(tags or [])))


def test_lethe_adapter_bounds_recall_and_persists_output() -> None:
    stub = MemoryStub()
    adapter = LetheMemoryAdapter(stub, recall_k=2)

    assert adapter.recall_text("current task") == ["bounded memory"]
    adapter.remember_output("answer", session_id="run-1")

    assert stub.saved == [("answer", "run-1", ["soteria_output"])]


def test_lethe_adapter_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="recall_k"):
        LetheMemoryAdapter(MemoryStub(), recall_k=0)
