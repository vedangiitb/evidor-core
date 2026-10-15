"""Sink abstractions and built-in sinks for Evidor telemetry."""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from .events import TelemetryEvent, redact_event

T_Event = TypeVar("T_Event", bound=TelemetryEvent)


@runtime_checkable
class TelemetrySink(Protocol):
    """Destination for exported telemetry events."""

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        """Process a batch of emitted telemetry events."""
        ...

    def flush(self) -> None:
        """Flush any buffered events."""
        ...

    def close(self) -> None:
        """Release any resources held by this sink."""
        ...


class InMemorySink:
    """Thread-safe in-memory sink for testing, inspection, and local assertions.

    Args:
        capture_content: When True (default), user prompts, model outputs, and tool
            arguments are captured. When False (metadata-only), all raw text payloads
            are redacted.
        capture_thinking: When True and capture_content is True, internal thinking
            scratchpads are captured. Note: capture_thinking cannot override
            capture_content=False (privacy boundary).
    """

    def __init__(self, capture_content: bool = True, capture_thinking: bool = False) -> None:
        self._capture_content = capture_content
        self._capture_thinking = capture_thinking
        self._events: list[TelemetryEvent] = []
        self._lock = threading.Lock()

    @property
    def capture_content(self) -> bool:
        return self._capture_content

    @property
    def capture_thinking(self) -> bool:
        return self._capture_thinking

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        with self._lock:
            to_store = [
                redact_event(e, capture_content=self._capture_content, capture_thinking=self._capture_thinking)
                for e in events
            ]
            self._events.extend(to_store)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def events(self) -> list[TelemetryEvent]:
        """Return a snapshot copy of all recorded events."""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()

    def filter(self, event_type: type[T_Event]) -> list[T_Event]:
        """Return all recorded events matching the specified type."""
        with self._lock:
            return [e for e in self._events if isinstance(e, event_type)]


class ConsoleSink:
    """Simple sink printing events to stderr or stdout for local debugging.

    Args:
        stream: Output file stream (defaults to sys.stderr).
        capture_content: When True (default), payload texts are printed.
        capture_thinking: When True and capture_content is True, internal thinking
            scratchpads are printed. Note: capture_thinking cannot override
            capture_content=False (privacy boundary).
    """

    def __init__(self, stream: Any = None, capture_content: bool = True, capture_thinking: bool = False) -> None:
        self._stream = stream or sys.stderr
        self._capture_content = capture_content
        self._capture_thinking = capture_thinking
        self._lock = threading.Lock()

    @property
    def capture_content(self) -> bool:
        return self._capture_content

    @property
    def capture_thinking(self) -> bool:
        return self._capture_thinking

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        with self._lock:
            for event in events:
                ev = redact_event(
                    event, capture_content=self._capture_content, capture_thinking=self._capture_thinking
                )
                attrs = ev.to_attributes()
                cls_name = ev.__class__.__name__
                trace_id = ev.trace_context.trace_id[:8]
                span_id = ev.trace_context.span_id[:8]
                print(f"[Telemetry][{trace_id}:{span_id}] {cls_name}: {attrs}", file=self._stream)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        self.flush()
