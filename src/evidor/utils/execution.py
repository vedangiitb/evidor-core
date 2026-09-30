"""Execution helpers for coroutine bridging and time-bounded callables."""

import asyncio
from collections.abc import Callable
import concurrent.futures
import contextvars
from typing import Any


def run_coroutine_sync(factory: Callable[[], Any]) -> Any:
    """Run a coroutine factory synchronously, preserving context inside active loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        context = contextvars.copy_context()

        def worker() -> Any:
            return context.run(asyncio.run, factory())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(worker).result()
    return asyncio.run(factory())


def run_with_timeout(func: Callable[[], Any], timeout: float | None, error_message: str) -> Any:
    """Execute a callable synchronously, enforcing a timeout if configured."""
    if timeout is not None and timeout > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as err:
                raise TimeoutError(error_message) from err
    return func()


async def run_with_timeout_async(coro: Any, timeout: float | None, error_message: str) -> Any:
    """Await a coroutine, enforcing a timeout if configured."""
    if timeout is not None and timeout > 0:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as err:
            raise TimeoutError(error_message) from err
    return await coro
