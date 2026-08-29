"""Tests for the schema registry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from soteria_loop.schemas import (
    SchemaError,
    SchemaRegistry,
    ValidationFailure,
)


class ReadFileArgs(BaseModel):
    path: str = Field(min_length=1)


class SearchArgs(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


def test_register_and_get() -> None:
    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    assert reg.get("read_file") is ReadFileArgs


def test_register_rejects_empty_name() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaError, match="name"):
        reg.register("", ReadFileArgs)


def test_register_rejects_non_basemodel() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaError, match="BaseModel"):
        reg.register("bad", dict)  # type: ignore[arg-type]


def test_register_rejects_duplicate() -> None:
    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    with pytest.raises(SchemaError, match="already registered"):
        reg.register("read_file", SearchArgs)


def test_get_unknown_raises() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaError, match="not registered"):
        reg.get("nope")


def test_has_returns_bool() -> None:
    reg = SchemaRegistry()
    reg.register("a", ReadFileArgs)
    assert reg.has("a") is True
    assert reg.has("b") is False


def test_names_returns_sorted_iterable() -> None:
    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    reg.register("search", SearchArgs)
    assert reg.names() == ("read_file", "search")


def test_unregister_returns_bool() -> None:
    reg = SchemaRegistry()
    reg.register("a", ReadFileArgs)
    assert reg.unregister("a") is True
    assert reg.unregister("a") is False


def test_validate_dict_payload() -> None:
    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    result = reg.validate("read_file", {"path": "/tmp/x"})
    assert isinstance(result, ReadFileArgs)
    assert result.path == "/tmp/x"


def test_validate_returns_existing_basemodel_unchanged() -> None:
    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    instance = ReadFileArgs(path="/tmp/x")
    result = reg.validate("read_file", instance)
    assert result is instance


def test_validate_rejects_invalid_payload() -> None:
    reg = SchemaRegistry()
    reg.register("search", SearchArgs)
    with pytest.raises(ValidationFailure, match="failed validation"):
        reg.validate("search", {"query": "", "limit": 0})


def test_validate_rejects_non_mapping_payload() -> None:
    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    with pytest.raises(ValidationFailure, match="must be a mapping"):
        reg.validate("read_file", "not a dict")


def test_validate_missing_schema_raises() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaError, match="not registered"):
        reg.validate("nope", {"k": "v"})


def test_merge_adds_schemas() -> None:
    a = SchemaRegistry()
    b = SchemaRegistry()
    a.register("read_file", ReadFileArgs)
    b.register("search", SearchArgs)
    added = a.merge(b)
    assert added == 1
    assert a.has("search")


def test_merge_skips_duplicates() -> None:
    a = SchemaRegistry()
    b = SchemaRegistry()
    a.register("read_file", ReadFileArgs)
    b.register("read_file", SearchArgs)
    with pytest.raises(SchemaError):
        a.merge(b)


def test_iter_and_contains() -> None:
    reg = SchemaRegistry()
    reg.register("a", ReadFileArgs)
    reg.register("b", SearchArgs)
    assert "a" in reg
    assert set(reg) == {"a", "b"}


def test_len_reports_count() -> None:
    reg = SchemaRegistry()
    assert len(reg) == 0
    reg.register("a", ReadFileArgs)
    reg.register("b", SearchArgs)
    assert len(reg) == 2


def test_validate_accepts_object_with_dict_method() -> None:
    class DictLike:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __iter__(self) -> Any:
            return iter(self._data.items())

    reg = SchemaRegistry()
    reg.register("read_file", ReadFileArgs)
    result = reg.validate("read_file", DictLike({"path": "/tmp/x"}))
    assert result.path == "/tmp/x"
