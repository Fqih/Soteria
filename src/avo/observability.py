"""OpenTelemetry integration for avo.

This module is a thin wrapper around the optional
``opentelemetry-api`` package. When the package is not installed or
when ``AVO_OTEL_ENABLED`` is unset, every helper falls back to a
no-op implementation so the rest of the runtime never has to branch
on observability state.

Activation:

- Install the optional extra: ``pip install avo[otel]``.
- Set ``AVO_OTEL_ENABLED=1`` in the environment.
- Configure an OTLP endpoint via ``OTEL_EXPORTER_OTLP_ENDPOINT``
  (defaults to ``http://localhost:4317``).

Span shape follows the OpenTelemetry semantic conventions for
generative AI systems (see
``https://opentelemetry.io/docs/specs/semconv/gen-ai/``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from avo.exceptions import AvoError

__all__ = [
    "OtelDisabledError",
    "configure_tracer",
    "is_enabled",
    "record_tool_call",
    "record_usage",
    "span_for_turn",
]


class OtelDisabledError(AvoError):
    """Raised when observability helpers are used without activation."""


_ENABLED: bool = False
_TRACER: Any = None  # opentelemetry.trace.Tracer or a no-op stub.
_PROVIDER: Any = None  # opentelemetry.sdk.trace.TracerProvider, optional.


def is_enabled() -> bool:
    """Return ``True`` when OpenTelemetry emission is wired up."""

    return _ENABLED


def _noop_tracer() -> Any:
    """Return a tracer whose spans are inert no-ops."""

    class _NoopSpan:
        def set_attribute(self, key: str, value: object) -> None:
            return None

        def set_status(self, *args: object, **kwargs: object) -> None:
            return None

        def record_exception(self, exc: BaseException) -> None:
            return None

        def end(self) -> None:
            return None

    class _NoopTracer:
        @contextmanager
        def start_as_current_span(
            self, name: str, attributes: dict[str, Any] | None = None
        ) -> Iterator[_NoopSpan]:
            del attributes
            yield _NoopSpan()

    return _NoopTracer()


def configure_tracer(*, service_name: str | None = None) -> Any:
    """Initialize the global tracer provider once per process.

    The function is idempotent: subsequent calls return the existing
    tracer without re-initializing the SDK. Set ``AVO_OTEL_ENABLED=1``
    in the environment before calling for real activation.
    """

    global _ENABLED, _TRACER, _PROVIDER

    if not (os.environ.get("AVO_OTEL_ENABLED") or "").strip():
        _ENABLED = False
        _TRACER = _noop_tracer()
        return _TRACER

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise OtelDisabledError(
            "AVO_OTEL_ENABLED=1 requires the [otel] extra. Install with `pip install avo[otel]`."
        ) from exc

    if _PROVIDER is None:
        resource = Resource.create(
            {
                "service.name": service_name or os.environ.get("AVO_OTEL_SERVICE_NAME", "avo"),
                "service.version": "0.1.2",
            }
        )
        provider = TracerProvider(resource=resource)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            # Fall back to in-memory only; exporters are optional.
            pass
        trace.set_tracer_provider(provider)
        _PROVIDER = provider

    _ENABLED = True
    _TRACER = trace.get_tracer("avo")
    return _TRACER


def span_for_turn(run_id: str, provider: str | None = None, model: str | None = None) -> Any:
    """Context manager that wraps one turn in an OpenTelemetry span.

    The span carries the standard ``gen_ai.*`` attributes:

    - ``gen_ai.system`` — provider name when known.
    - ``gen_ai.request.model`` — model identifier when known.
    - ``avo.run_id`` — local run identifier.

    When observability is disabled the helper returns a no-op context
    manager so callers can use it unconditionally.
    """

    if _TRACER is None:
        configure_tracer()
    assert _TRACER is not None

    attributes: dict[str, str] = {"avo.run_id": run_id}
    if provider:
        attributes["gen_ai.system"] = provider
    if model:
        attributes["gen_ai.request.model"] = model
    return _TRACER.start_as_current_span("avo.turn", attributes=attributes)


def record_tool_call(span: Any, name: str, *, call_id: str | None = None) -> None:
    """Annotate the active span with a tool-call event.

    Safe to call on the no-op span; the call is simply ignored.
    """

    if span is None:
        return
    span.set_attribute("avo.tool.name", name)
    if call_id:
        span.set_attribute("avo.tool.call_id", call_id)


def record_usage(
    span: Any,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> None:
    """Annotate the active span with token usage and cost attributes."""

    if span is None:
        return
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if cost_usd is not None:
        span.set_attribute("avo.cost_usd", cost_usd)
