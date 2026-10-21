"""Shared client lifecycle and event loop management for model providers."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from .base import is_client_closed


def close_sync_client(client: Any) -> None:
    """Safely close a synchronous SDK client."""
    if client is not None:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


async def close_async_clients(
    clients_map: dict[asyncio.AbstractEventLoop, Any],
    default_client: Any = None,
    lock: threading.Lock | None = None,
) -> None:
    """Safely close async clients across all registered event loops.

    Handles cross-thread event loop execution if loops are still running in other threads.
    """
    current_loop = None
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if lock is not None:
        with lock:
            clients_to_close = list(clients_map.items())
            clients_map.clear()
    else:
        clients_to_close = list(clients_map.items())
        clients_map.clear()

    for loop, client in clients_to_close:
        try:
            if loop is current_loop:
                await client.close()
            elif loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(client.close(), loop)
                await asyncio.wrap_future(fut)
            else:
                await client.close()
        except Exception:
            pass

    if default_client is not None:
        try:
            await default_client.close()
        except Exception:
            pass


def get_or_create_async_client(
    clients_map: dict[asyncio.AbstractEventLoop, Any],
    current_default: Any,
    factory: Callable[[], Any],
    lock: threading.Lock,
) -> tuple[Any, Any]:
    """Get or create an async client for the current event loop, pruning closed loops.

    Returns:
        tuple: (active_client, updated_default_client)
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    with lock:
        # Prune closed loops
        dead_loops = [l for l in clients_map if l.is_closed()]
        for l in dead_loops:
            del clients_map[l]

        if loop is not None:
            client = clients_map.get(loop)
            if is_client_closed(client):
                client = factory()
                clients_map[loop] = client
            return client, current_default
        else:
            if is_client_closed(current_default):
                current_default = factory()
            return current_default, current_default
