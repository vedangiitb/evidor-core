"""Actor-like telemetry runtime with a bounded queue and background consumer."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from typing import Any, Literal

from .events import TelemetryEvent
from .sink import TelemetrySink

OverflowStrategy = Literal["drop_newest", "drop_oldest"]
_STOP_SENTINEL = object()


class TelemetryRuntime:
    """Manages event buffering on a bounded queue and asynchronous background dispatch."""

    def __init__(
        self,
        sinks: Sequence[TelemetrySink] | None = None,
        *,
        queue_size: int = 1000,
        batch_size: int = 50,
        flush_interval_seconds: float = 0.2,
        overflow_strategy: OverflowStrategy = "drop_newest",
    ) -> None:
        self._sinks = list(sinks or ())
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._overflow_strategy: OverflowStrategy = overflow_strategy

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._worker_thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._closed = False
        self._lock = threading.Lock()
        self._condition = threading.Condition()
        self._pending_count = 0
        self._dropped_count = 0

    @property
    def dropped_count(self) -> int:
        """The number of events dropped due to queue overflow."""
        return self._dropped_count

    @property
    def sinks(self) -> tuple[TelemetrySink, ...]:
        """Configured telemetry sinks."""
        return tuple(self._sinks)

    def add_sink(self, sink: TelemetrySink) -> None:
        """Register an additional telemetry sink."""
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def _ensure_worker_started(self) -> None:
        if self._started or self._closed or self._stopping:
            return
        with self._lock:
            if self._started or self._closed or self._stopping:
                return
            self._started = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="Evidor-Telemetry-Worker",
                daemon=True,
            )
            self._worker_thread.start()

    def emit(self, event: TelemetryEvent) -> None:
        """Enqueue an event non-blockingly (< 1µs)."""
        if self._closed or self._stopping or not self._sinks:
            return

        self._ensure_worker_started()

        try:
            self._queue.put_nowait(event)
            with self._condition:
                self._pending_count += 1
        except queue.Full:
            self._dropped_count += 1
            if self._overflow_strategy == "drop_oldest":
                try:
                    self._queue.get_nowait()
                    with self._condition:
                        self._pending_count -= 1
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(event)
                    with self._condition:
                        self._pending_count += 1
                except queue.Full:
                    pass

    def _worker_loop(self) -> None:
        stop_seen = False
        while not stop_seen:
            batch: list[TelemetryEvent] = []
            try:
                item = self._queue.get(timeout=self._flush_interval)
                if item is _STOP_SENTINEL:
                    break
                batch.append(item)
                while len(batch) < self._batch_size:
                    try:
                        next_item = self._queue.get_nowait()
                        if next_item is _STOP_SENTINEL:
                            stop_seen = True
                            break
                        batch.append(next_item)
                    except queue.Empty:
                        break
            except queue.Empty:
                if self._stopping and self._queue.empty():
                    break

            if batch:
                try:
                    self._dispatch_batch(batch)
                finally:
                    with self._condition:
                        self._pending_count -= len(batch)
                        if self._pending_count <= 0:
                            self._pending_count = 0
                            self._condition.notify_all()

    def _dispatch_batch(self, batch: Sequence[TelemetryEvent]) -> None:
        for sink in list(self._sinks):
            try:
                sink.write(batch)
            except Exception:
                # Prevent any faulty sink from terminating the telemetry runtime
                pass

    def flush(self, timeout: float = 2.0) -> None:
        """Block until all queued and in-flight events are dispatched and flush all sinks."""
        deadline = time.time() + timeout
        with self._condition:
            while self._pending_count > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

        with self._lock:
            for sink in self._sinks:
                try:
                    sink.flush()
                except Exception:
                    pass

    def close(self, timeout: float = 2.0) -> None:
        """Stop the background consumer, drain remaining events, and release sink resources."""
        with self._lock:
            if self._closed or self._stopping:
                return
            self._stopping = True

        if self._started:
            try:
                self._queue.put_nowait(_STOP_SENTINEL)
            except queue.Full:
                # Queue is full; stopping flag ensures worker drains all events and terminates
                pass

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)

        # Final state transition and resource release
        with self._lock:
            self._closed = True
            for sink in self._sinks:
                try:
                    sink.flush()
                except Exception:
                    pass
                try:
                    sink.close()
                except Exception:
                    pass

    def __enter__(self) -> TelemetryRuntime:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
