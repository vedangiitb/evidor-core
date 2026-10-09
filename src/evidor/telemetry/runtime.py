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
        self._closed = False
        self._lock = threading.Lock()
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
        if self._started or self._closed:
            return
        with self._lock:
            if self._started or self._closed:
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
        if self._closed or not self._sinks:
            return

        self._ensure_worker_started()

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_count += 1
            if self._overflow_strategy == "drop_oldest":
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(event)
                except queue.Full:
                    pass

    def _worker_loop(self) -> None:
        while not self._closed:
            batch: list[TelemetryEvent] = []
            try:
                item = self._queue.get(timeout=self._flush_interval)
                if item is _STOP_SENTINEL:
                    self._queue.task_done()
                    break
                batch.append(item)
                while len(batch) < self._batch_size:
                    try:
                        next_item = self._queue.get_nowait()
                        if next_item is _STOP_SENTINEL:
                            self._queue.task_done()
                            self._closed = True
                            break
                        batch.append(next_item)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            if batch:
                self._dispatch_batch(batch)
                for _ in batch:
                    self._queue.task_done()

    def _dispatch_batch(self, batch: Sequence[TelemetryEvent]) -> None:
        for sink in list(self._sinks):
            try:
                sink.write(batch)
            except Exception:
                # Prevent any faulty sink from terminating the telemetry runtime
                pass

    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue is fully drained and flush all sinks."""
        deadline = time.time() + timeout

        # Process any pending items if worker hasn't processed them yet
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.01)

        with self._lock:
            for sink in self._sinks:
                try:
                    sink.flush()
                except Exception:
                    pass

    def close(self, timeout: float = 2.0) -> None:
        """Stop the background consumer and release sink resources."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if self._started:
            try:
                self._queue.put_nowait(_STOP_SENTINEL)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(_STOP_SENTINEL)
                except Exception:
                    pass

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)

        # Final flush & close sinks
        with self._lock:
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
