"""Execution helpers for coroutine bridging and time-bounded callables."""

import asyncio
from typing import Any


def check_sync_allowed(error_message: str | None = None) -> None:
    """Raise RuntimeError if called within an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    msg = error_message or (
        "This sync method cannot be called from a running event loop. "
        "Use its async equivalent instead."
    )
    raise RuntimeError(msg)


class BackgroundEventLoop:
    """Manages a persistent asyncio event loop running on a dedicated background daemon thread.

    Provides coroutine bridging across repeated synchronous calls without destroying
    underlying connection pools, SSL contexts, or event loops between invocations.
    Compatible with Python 3.10+.
    """

    def __init__(self, name: str = "EvidorBackgroundLoop") -> None:
        import contextvars
        import threading

        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def is_running(self) -> bool:
        """Return True if the background loop thread is active and running."""
        with self._lock:
            return self._loop is not None and self._loop.is_running() and not self._closed

    def _ensure_running(self) -> asyncio.AbstractEventLoop:
        import threading

        with self._lock:
            if self._closed:
                raise RuntimeError("BackgroundEventLoop has been closed.")

            if self._loop is not None and self._loop.is_running():
                return self._loop

            loop = asyncio.new_event_loop()
            started_event = threading.Event()

            def _thread_target() -> None:
                asyncio.set_event_loop(loop)
                started_event.set()
                try:
                    loop.run_forever()
                finally:
                    try:
                        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                    except Exception:
                        pass
                    try:
                        loop.close()
                    except Exception:
                        pass

            thread = threading.Thread(
                target=_thread_target,
                name=f"{self._name}-{threading.get_ident()}",
                daemon=True,
            )
            thread.start()
            started_event.wait()

            self._loop = loop
            self._thread = thread
            return self._loop

    def run(
        self,
        target: Any,
        error_message: str | None = None,
    ) -> Any:
        """Run an async coroutine or coroutine factory synchronously on the background loop.

        Raises RuntimeError if called from a running event loop on the current thread.
        """
        import contextvars

        check_sync_allowed(error_message)
        loop = self._ensure_running()

        ctx = contextvars.copy_context()

        async def _entrypoint() -> Any:
            coro = target() if callable(target) else target
            return await coro

        future = asyncio.run_coroutine_threadsafe(ctx.run(_entrypoint), loop)
        return future.result()

    def close(self) -> None:
        """Stop the background event loop and join the worker thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)


def run_async_in_sync(
    target: Any,
    error_message: str | None = None,
) -> Any:
    """Execute an async coroutine or coroutine factory synchronously.

    Raises RuntimeError if called within an active event loop.
    """
    check_sync_allowed(error_message)
    coro = target() if callable(target) else target
    return asyncio.run(coro)


async def run_with_timeout_async(coro: Any, timeout: float | None, error_message: str) -> Any:
    """Await a coroutine, enforcing a timeout if configured."""
    if timeout is not None and timeout > 0:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as err:
            raise TimeoutError(error_message) from err
    return await coro
