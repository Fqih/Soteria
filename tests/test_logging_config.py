"""Tests for :mod:`avo.logging_config`."""

from __future__ import annotations

import io
import json
import logging

import pytest

from avo.logging_config import (
    JsonFormatter,
    configure_logging,
    install_json_handler,
)


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def json_logger(stream: io.StringIO) -> logging.Logger:
    logger = logging.getLogger("avo.test.json")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel("DEBUG")
    logger.propagate = False
    return logger


def _parse(stream: io.StringIO) -> dict[str, object]:
    line = stream.getvalue().strip().splitlines()[-1]
    return json.loads(line)


def test_json_formatter_emits_required_fields(
    json_logger: logging.Logger, stream: io.StringIO
) -> None:
    json_logger.warning("hello world")
    record = _parse(stream)
    assert record["level"] == "WARNING"
    assert record["logger"] == "avo.test.json"
    assert record["message"] == "hello world"
    assert "ts" in record
    # ts is RFC 3339 — has the date, time, and timezone offset
    assert "T" in str(record["ts"])
    assert "+" in str(record["ts"]) or "Z" in str(record["ts"])


def test_json_formatter_surfaces_extras(json_logger: logging.Logger, stream: io.StringIO) -> None:
    json_logger.info("event", extra={"run_id": "run-42", "step": 3})
    record = _parse(stream)
    assert record["run_id"] == "run-42"
    assert record["step"] == 3


def test_json_formatter_stringifies_non_serializable_extras(
    json_logger: logging.Logger, stream: io.StringIO
) -> None:
    class _Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    json_logger.info("payload", extra={"obj": _Opaque()})
    record = _parse(stream)
    assert record["obj"] == "<opaque>"


def test_json_formatter_captures_exception(
    json_logger: logging.Logger, stream: io.StringIO
) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        json_logger.exception("failed")
    record = _parse(stream)
    assert "exc_info" in record
    assert "ValueError: boom" in str(record["exc_info"])


def test_install_json_handler_attaches_formatter() -> None:
    logger = logging.getLogger("avo.test.install")
    logger.handlers.clear()
    handler = install_json_handler(logger, level="INFO")
    try:
        assert isinstance(handler.formatter, JsonFormatter)
        assert logger.level == logging.INFO
    finally:
        logger.removeHandler(handler)


def test_configure_logging_json_mode_returns_handler() -> None:
    logger = logging.getLogger("avo.test.configure")
    logger.handlers.clear()
    handler = configure_logging(level="INFO", json_mode=True, logger_name="avo.test.configure")
    try:
        assert handler is not None
        assert isinstance(handler.formatter, JsonFormatter)
    finally:
        logger.removeHandler(handler)


def test_configure_logging_text_mode_returns_none() -> None:
    logger = logging.getLogger("avo.test.text")
    logger.handlers.clear()
    handler = configure_logging(level="INFO", json_mode=False, logger_name="avo.test.text")
    try:
        assert handler is None
    finally:
        for existing in list(logger.handlers):
            logger.removeHandler(existing)


def test_configure_logging_json_mode_is_idempotent() -> None:
    logger_name = "avo.test.idempotent"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    configure_logging(level="INFO", json_mode=True, logger_name=logger_name)
    configure_logging(level="INFO", json_mode=True, logger_name=logger_name)
    try:
        json_handlers = [h for h in logger.handlers if isinstance(h.formatter, JsonFormatter)]
        assert len(json_handlers) == 1
    finally:
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
