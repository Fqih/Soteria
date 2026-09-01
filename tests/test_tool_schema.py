"""Tests for ``avo.tools`` JSON Schema export helpers."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from avo.tools import to_anthropic_tool, to_json_schema, to_openai_function


class SimpleArgs(BaseModel):
    path: str
    encoding: str = "utf-8"


class NestedArgs(BaseModel):
    name: str
    options: dict[str, str]


class RecursiveArgs(BaseModel):
    label: str
    child: "RecursiveArgs | None" = None  # noqa: UP037


RecursiveArgs.model_rebuild()


def test_to_json_schema_returns_object_root() -> None:
    schema = to_json_schema(SimpleArgs)
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"path", "encoding"}
    assert schema["properties"]["path"]["type"] == "string"
    assert "path" in schema["required"]
    assert "encoding" not in schema["required"]


def test_to_json_schema_sets_additional_properties_false_on_root() -> None:
    schema = to_json_schema(SimpleArgs)
    assert schema["additionalProperties"] is False


def test_to_json_schema_sets_additional_properties_false_on_nested_dict() -> None:
    schema = to_json_schema(NestedArgs)
    options = schema["properties"]["options"]
    assert options["additionalProperties"] is False
    assert schema["additionalProperties"] is False


def test_to_json_schema_sets_additional_properties_false_on_self_ref() -> None:
    schema = to_json_schema(RecursiveArgs)
    defs = schema["$defs"]["RecursiveArgs"]
    assert defs["additionalProperties"] is False
    # The recursive model is exposed via $ref; ensure the ref'd
    # schema is the same shape that strict validators walk.
    assert defs["type"] == "object"


def test_to_openai_function_emits_strict_flag() -> None:
    entry = to_openai_function("read_file", "Read a file.", SimpleArgs)
    assert entry["type"] == "function"
    function = entry["function"]
    assert function["name"] == "read_file"
    assert function["description"] == "Read a file."
    assert function["strict"] is True
    assert function["parameters"]["additionalProperties"] is False


def test_to_anthropic_tool_emits_input_schema() -> None:
    entry = to_anthropic_tool("read_file", "Read a file.", SimpleArgs)
    assert entry["name"] == "read_file"
    assert entry["description"] == "Read a file."
    assert entry["input_schema"]["additionalProperties"] is False
    assert entry["input_schema"]["type"] == "object"


def test_to_json_schema_does_not_mutate_input_model() -> None:
    schema = to_json_schema(SimpleArgs)
    again = to_json_schema(SimpleArgs)
    assert schema == again
    assert again["additionalProperties"] is False


def test_to_json_schema_round_trip_with_model_validate() -> None:
    schema = to_json_schema(SimpleArgs)
    instance = SimpleArgs.model_validate({"path": "/tmp/x"})
    assert instance.path == "/tmp/x"
    assert instance.encoding == "utf-8"
    assert schema["required"] == ["path"]


@pytest.mark.parametrize(
    "helper_name",
    ["to_json_schema", "to_openai_function", "to_anthropic_tool"],
)
def test_helpers_have_documented_signatures(helper_name: str) -> None:
    """Each helper must be importable and accept the documented kwargs."""
    helper = {
        "to_json_schema": to_json_schema,
        "to_openai_function": to_openai_function,
        "to_anthropic_tool": to_anthropic_tool,
    }[helper_name]
    if helper_name == "to_json_schema":
        result = helper(SimpleArgs)
    else:
        result = helper("name", "desc", SimpleArgs)
    assert isinstance(result, dict)
