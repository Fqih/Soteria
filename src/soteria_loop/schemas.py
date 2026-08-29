"""Schema registry — keyed Pydantic BaseModel store.

Tools / providers register their argument models under a stable name,
and the runtime validates incoming payloads against the model. The
registry is a thin wrapper around a dict; the heavy lifting is done
by Pydantic itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from soteria_loop.exceptions import SoteriaError

SchemaError = SoteriaError
ValidationFailure = SoteriaError


class SchemaRegistry:
    """Registry of ``name -> BaseModel``."""

    __slots__ = ("_schemas",)

    def __init__(self) -> None:
        self._schemas: dict[str, type[BaseModel]] = {}

    def register(self, name: str, model: type[BaseModel]) -> None:
        if not name:
            raise SchemaError("schema name must be non-empty")
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise SchemaError(f"schema for {name!r} must be a BaseModel subclass")
        if name in self._schemas:
            raise SchemaError(f"schema {name!r} already registered")
        self._schemas[name] = model

    def unregister(self, name: str) -> bool:
        return self._schemas.pop(name, None) is not None

    def get(self, name: str) -> type[BaseModel]:
        try:
            return self._schemas[name]
        except KeyError as exc:
            raise SchemaError(f"schema {name!r} not registered") from exc

    def has(self, name: str) -> bool:
        return name in self._schemas

    def names(self) -> tuple[str, ...]:
        return tuple(self._schemas.keys())

    def validate(self, name: str, payload: Mapping[str, Any] | Any) -> BaseModel:
        """Validate ``payload`` against the registered schema."""

        schema = self.get(name)
        if isinstance(payload, BaseModel):
            return payload
        if isinstance(payload, dict):
            data = cast("Mapping[str, Any]", payload)
        else:
            try:
                data = cast("Mapping[str, Any]", dict(payload))
            except (TypeError, ValueError) as exc:
                raise ValidationFailure(
                    f"payload for {name!r} must be a mapping, got {type(payload).__name__}"
                ) from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValidationFailure(f"payload for {name!r} failed validation: {exc}") from exc

    def merge(self, other: SchemaRegistry) -> int:
        """Register all schemas from ``other``. Returns count added."""

        added = 0
        for name in other.names():
            self.register(name, other.get(name))
            added += 1
        return added

    def __iter__(self) -> Iterable[str]:
        return iter(self._schemas)

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)


__all__ = [
    "SchemaError",
    "SchemaRegistry",
    "ValidationFailure",
]
