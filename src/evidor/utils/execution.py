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
