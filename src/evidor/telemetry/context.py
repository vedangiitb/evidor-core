"""Trace context propagation and telemetry runtime scoping via contextvars."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from .events import TelemetryEvent, TraceContext

if TYPE_CHECKING:
    from .runtime import TelemetryRuntime

_CURRENT_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar("evidor_trace_context", default=None)
_CURRENT_TELEMETRY: ContextVar[Any] = ContextVar("evidor_telemetry_runtime", default=None)


def get_current_trace_context() -> TraceContext | None:
    """Retrieve the currently active trace context in this execution context."""
    return _CURRENT_TRACE_CONTEXT.get()


@contextmanager
def trace_scope(ctx: TraceContext) -> Iterator[TraceContext]:
    """Establish an active trace context for the duration of the context block."""
    token = _CURRENT_TRACE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_TRACE_CONTEXT.reset(token)


def create_child_trace_context() -> TraceContext:
    """Derive a child trace context from the active context, or generate a fresh root context."""
    current = get_current_trace_context()
    if current is not None:
        return current.child()
    return TraceContext.new_root()


def get_active_telemetry() -> TelemetryRuntime | None:
    """Retrieve the currently active TelemetryRuntime in this execution context."""
    return _CURRENT_TELEMETRY.get()


@contextmanager
def telemetry_scope(runtime: TelemetryRuntime | None) -> Iterator[TelemetryRuntime | None]:
    """Establish an active TelemetryRuntime for the duration of the context block."""
    token = _CURRENT_TELEMETRY.set(runtime)
    try:
        yield runtime
    finally:
        _CURRENT_TELEMETRY.reset(token)


def emit_event(event: TelemetryEvent) -> None:
    """Emit a telemetry event to the active TelemetryRuntime, if one is configured."""
    runtime = get_active_telemetry()
    if runtime is not None:
        runtime.emit(event)

