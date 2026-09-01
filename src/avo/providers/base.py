"""Provider-neutral model protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from avo.models import ModelRequest, ModelResponse


@runtime_checkable
class ModelProvider(Protocol):
    """An async provider capable of producing one agent-loop decision."""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a final answer or one tool call."""


@runtime_checkable
class StatefulModelProvider(Protocol):
    """Optional checkpoint hooks implemented by deterministic providers."""

    def snapshot_state(self) -> dict[str, JsonValue]:
        """Return JSON-safe provider state for a checkpoint."""

    def restore_state(self, state: dict[str, JsonValue]) -> None:
        """Restore provider state from a checkpoint."""
