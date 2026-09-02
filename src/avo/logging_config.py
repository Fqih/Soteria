"""Structured logging configuration.

Two output modes:

* ``text`` (default) — the stdlib ``Formatter`` style, easy to read in
  a terminal.
* ``json`` — one JSON object per line, suitable for shipping into
  log aggregators (Loki, Datadog, ELK, etc.). Activate with
  ``AVO_LOG_FORMAT=json`` or by calling
  :func:`configure_logging` with ``json_mode=True``.

The JSON formatter carries:

* ``ts`` — RFC 3339 UTC timestamp
* ``level`` — log level name
* ``logger`` — logger name
* ``message`` — formatted message
* ``exc_info`` — stringified traceback when ``exc_info`` is set
* any ``extra=`` keys passed to the logger

Unknown ``LogRecord`` internals (``args``, ``msg``, ``levelno``,
``pathname``, ``lineno``, ``funcName``, ``created``, ``msecs``,
``relativeCreated``, ``thread``, ``threadName``, ``processName``,
``process``, ``name``, ``stack_info``) are dropped to keep the
output deterministic across Python versions.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Final, Literal

__all__ = ["JsonFormatter", "configure_logging", "install_json_handler"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]

_STDLIB_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Format :class:`logging.LogRecord` instances as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Surface any ``extra={...}`` keys supplied at the call site.
        for key, value in record.__dict__.items():
            if key in _STDLIB_RECORD_KEYS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def install_json_handler(
    logger: logging.Logger,
    *,
    level: LogLevel = "INFO",
) -> logging.Handler:
    """Attach a stdout :class:`logging.StreamHandler` with :class:`JsonFormatter`."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


def configure_logging(
    *,
    level: LogLevel = "INFO",
    json_mode: bool = False,
    logger_name: str | None = None,
) -> logging.Handler | None:
    """Configure the named logger (or root if ``None``).

    Returns the JSON handler when ``json_mode`` is set, otherwise
    ``None``. The handler is replaced on each call so callers can
    swap modes at runtime (e.g. ``avo chat`` switches to JSON when
    ``AVO_LOG_FORMAT=json`` is set).
    """

    logger = logging.getLogger(logger_name)
    for existing in list(logger.handlers):
        if isinstance(existing.formatter, JsonFormatter):
            logger.removeHandler(existing)
    if json_mode:
        return install_json_handler(logger, level=level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    return None
