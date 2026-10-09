"""OpenTelemetry adapter converting Evidor lifecycle events into distributed trace spans.

Provides interoperability with OpenTelemetry and OpenInference conventions
without coupling Evidor core to any external telemetry SDKs.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Any

from ..events import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    MCPCallEndEvent,
    MCPCallStartEvent,
    TelemetryEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    redact_event,
)
from ..sink import TelemetrySink

try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        NonRecordingSpan,
        Span,
        SpanContext,
        StatusCode,
        TraceFlags,
        Tracer,
        set_span_in_context,
    )

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    trace = None  # type: ignore[assignment]
    Span = Any  # type: ignore[assignment, misc]
    Tracer = Any  # type: ignore[assignment, misc]


def _sanitize_attribute(value: Any) -> str | bool | int | float | list[str | bool | int | float]:
    """Ensure attribute values conform to OpenTelemetry primitive types."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (str, bool, int, float)) for v in value):
            return list(value)
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


class OpenTelemetrySink(TelemetrySink):
    """Translates Evidor domain events into OpenTelemetry spans."""

    def __init__(
        self,
        tracer_provider: Any = None,
        tracer: Tracer | None = None,
        tracer_name: str = "evidor",
        capture_content: bool = True,
    ) -> None:
        if not HAS_OPENTELEMETRY:
            raise ImportError(
                "The 'opentelemetry-api' and 'opentelemetry-sdk' packages are required for the OpenTelemetry adapter. "
                "Install it with: pip install 'evidor[otel]'"
            )

        self._tracer_provider = tracer_provider
        self._capture_content = capture_content
        if tracer is not None:
            self._tracer: Tracer = tracer
        elif tracer_provider is not None:
            self._tracer = tracer_provider.get_tracer(tracer_name)
        else:
            self._tracer = trace.get_tracer(tracer_name)

        self._active_spans: dict[str, Span] = {}
        self._lock = threading.Lock()

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        """Process a batch of emitted events and update OpenTelemetry spans."""
        for event in events:
            ev = event if self._capture_content else redact_event(event)
            self._handle_event(ev)

    def _handle_event(self, event: TelemetryEvent) -> None:
        if isinstance(event, (AgentRunStartEvent, LLMCallStartEvent, ToolCallStartEvent, MCPCallStartEvent)):
            self._start_span(event)
        elif isinstance(event, (AgentRunEndEvent, LLMCallEndEvent, ToolCallEndEvent, MCPCallEndEvent)):
            self._end_span(event)
        elif isinstance(event, ErrorEvent):
            self._record_error(event)

    def _resolve_span_name(self, event: TelemetryEvent) -> str:
        if isinstance(event, (AgentRunStartEvent, AgentRunEndEvent)):
            return "agent.run"
        if isinstance(event, (LLMCallStartEvent, LLMCallEndEvent)):
            return f"llm.{event.model}" if event.model else "llm.call"
        if isinstance(event, (ToolCallStartEvent, ToolCallEndEvent)):
            return f"tool.{event.tool_name}" if event.tool_name else "tool.call"
        if isinstance(event, (MCPCallStartEvent, MCPCallEndEvent)):
            return f"mcp.{event.server_name}.{event.tool_name}"
        return "evidor.operation"

    def _build_parent_context(self, event: TelemetryEvent) -> Any:
        ctx = event.trace_context
        if not ctx.parent_span_id:
            return None
        with self._lock:
            parent_span = self._active_spans.get(ctx.parent_span_id)
        if parent_span is not None:
            return set_span_in_context(parent_span)
        try:
            trace_id_int = int(ctx.trace_id, 16)
            parent_span_id_int = int(ctx.parent_span_id, 16)
            parent_span_context = SpanContext(
                trace_id=trace_id_int,
                span_id=parent_span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
            )
            return set_span_in_context(NonRecordingSpan(parent_span_context))
        except (ValueError, TypeError):
            return None

    def _start_span(self, event: TelemetryEvent) -> None:
        name = self._resolve_span_name(event)
        parent_ctx = self._build_parent_context(event)
        start_time_ns = int(event.timestamp * 1e9)

        span = self._tracer.start_span(
            name=name,
            context=parent_ctx,
            start_time=start_time_ns,
        )

        # Set initial attributes
        for key, val in event.to_attributes().items():
            if val is not None:
                span.set_attribute(key, _sanitize_attribute(val))

        with self._lock:
            self._active_spans[event.trace_context.span_id] = span

    def _end_span(self, event: TelemetryEvent) -> None:
        with self._lock:
            span = self._active_spans.pop(event.trace_context.span_id, None)

        end_time_ns = int(event.timestamp * 1e9)

        if span is None:
            # Reconstruct completed span if StartEvent was dropped or missing
            name = self._resolve_span_name(event)
            parent_ctx = self._build_parent_context(event)
            duration_ms = getattr(event, "duration_ms", 0.0)
            start_time_ns = max(0, int((event.timestamp - (duration_ms / 1000.0)) * 1e9))
            span = self._tracer.start_span(
                name=name,
                context=parent_ctx,
                start_time=start_time_ns,
            )

        # Update end attributes
        for key, val in event.to_attributes().items():
            if val is not None:
                span.set_attribute(key, _sanitize_attribute(val))

        # Set span status
        status = getattr(event, "status", "ok")
        if status == "error":
            error_desc = getattr(event, "error", None) or "Error"
            span.set_status(StatusCode.ERROR, description=error_desc)
        elif status == "ok":
            span.set_status(StatusCode.OK)

        span.end(end_time=end_time_ns)

    def _record_error(self, event: ErrorEvent) -> None:
        with self._lock:
            span = self._active_spans.get(event.trace_context.span_id)

        if span is not None:
            attrs = {
                "exception.type": event.error_type,
                "exception.message": event.message,
            }
            for k, v in event.details.items():
                attrs[k] = _sanitize_attribute(v)
            span.add_event("exception", attributes=attrs)

    def flush(self) -> None:
        """Flush the underlying TracerProvider if it supports it."""
        if self._tracer_provider is not None and hasattr(self._tracer_provider, "force_flush"):
            try:
                self._tracer_provider.force_flush()
            except Exception:
                pass

    def close(self) -> None:
        """Shut down the underlying TracerProvider if it supports it."""
        self.flush()
        if self._tracer_provider is not None and hasattr(self._tracer_provider, "shutdown"):
            try:
                self._tracer_provider.shutdown()
            except Exception:
                pass
