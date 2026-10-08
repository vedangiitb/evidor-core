"""Background event loop runner for synchronous execution of async MCP sessions."""

import asyncio
from collections.abc import Coroutine
import threading
from typing import Any, TypeVar

T = TypeVar("T")


class BackgroundLoopRunner:
    """Runs a dedicated background thread hosting an asyncio event loop.

    Allows synchronous code to schedule and await coroutines safely on a long-lived loop.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="Evidor-MCP-Runner")
        self._thread.start()
        self._ready.wait()
        self._closed = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def run_coroutine(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Submit a coroutine to the background loop and wait synchronously for its result."""
        if self._closed:
            raise RuntimeError("BackgroundLoopRunner is closed.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the event loop and wait for the thread to exit."""
        if self._closed:
            return
        self._closed = True
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "BackgroundLoopRunner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
