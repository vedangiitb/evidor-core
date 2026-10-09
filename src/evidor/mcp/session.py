"""MCP session managing connection, lifecycle, and tool conversion for a single MCP server."""

import asyncio
from collections.abc import Sequence
import concurrent.futures
import json
from pathlib import Path
import threading
import time
from typing import Any

from evidor.tools import Tool
from ..telemetry import (
    ErrorEvent,
    MCPCallEndEvent,
    MCPCallStartEvent,
    create_child_trace_context,
    emit_event,
    trace_scope,
)

from .config import MCPServerConfig


def _serialize_mcp_result(result: Any) -> str:
    """Format and serialize CallToolResult into a string suitable for agent consumption."""
    if isinstance(result, str):
        return result

    is_error = getattr(result, "is_error", False) or getattr(result, "isError", False)
    content = getattr(result, "content", None)
    parts: list[str] = []

    if content:
        for item in content:
            if hasattr(item, "text"):
                parts.append(str(item.text))
            elif hasattr(item, "data"):
                mime = getattr(item, "mimeType", getattr(item, "mime_type", "unknown"))
                parts.append(f"[Binary data: {mime}]")
            elif hasattr(item, "resource"):
                res = item.resource
                uri = getattr(res, "uri", "resource")
                text = getattr(res, "text", str(res))
                parts.append(f"[Resource {uri}]: {text}")
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        try:
            parts.append(json.dumps(structured))
        except Exception:
            parts.append(str(structured))

    text_output = "\n".join(parts).strip()
    if not text_output:
        text_output = "Tool executed successfully with no output."

    if is_error:
        raise RuntimeError(f"MCP tool error: {text_output}")

    return text_output


class MCPSession:
    """Manages an active connection to a single MCP server."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.config = config
        self._owns_loop = loop is None
        self._loop = loop
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._worker_task: asyncio.Task | None = None
        self._cmd_queue: asyncio.Queue | None = None
        self._connected = False
        self._cached_tools: tuple[Tool, ...] | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop

        loop = asyncio.new_event_loop()
        self._loop = loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            self._ready_event.set()
            loop.run_forever()

        self._thread = threading.Thread(
            target=_run, daemon=True, name=f"Evidor-MCP-{self.config.name}"
        )
        self._thread.start()
        self._ready_event.wait()
        return self._loop

    async def _run_worker(
        self, started_future: concurrent.futures.Future[bool]
    ) -> None:
        try:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.sse import sse_client
            from mcp.client.streamable_http import streamable_http_client
            import httpx2
        except ImportError as exc:
            err = ImportError(
                "The 'mcp' package is required for MCP support. "
                "Install it with: pip install 'evidor[mcp]'"
            )
            started_future.set_exception(err)
            return

        cmd_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._cmd_queue = cmd_queue
        http_client = None

        try:
            if self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError(f"Server '{self.config.name}' requires a command.")
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                    cwd=str(self.config.cwd) if self.config.cwd is not None else None,
                    encoding=self.config.encoding,
                )
                cm = stdio_client(params)
            elif self.config.transport in ("streamable-http", "http"):
                if not self.config.url:
                    raise ValueError(f"Server '{self.config.name}' requires a url.")
                http_client = httpx2.AsyncClient(
                    headers=self.config.headers, timeout=self.config.timeout
                )
                cm = streamable_http_client(self.config.url, http_client=http_client)
            else:
                if not self.config.url:
                    raise ValueError(f"Server '{self.config.name}' requires a url.")
                cm = sse_client(
                    self.config.url,
                    headers=self.config.headers,
                    timeout=self.config.timeout,
                    sse_read_timeout=self.config.sse_read_timeout,
                )

            try:
                async with cm as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._connected = True
                        started_future.set_result(True)

                        while True:
                            msg = await cmd_queue.get()
                            if msg is None:
                                break
                            action, args, kwargs, result_future = msg
                            try:
                                func = getattr(session, action)
                                res = await func(*args, **kwargs)
                                result_future.set_result(res)
                            except Exception as err:
                                result_future.set_exception(err)
            finally:
                if http_client is not None:
                    await http_client.aclose()
        except Exception as exc:
            self._connected = False
            if not started_future.done():
                started_future.set_exception(exc)
            raise
        finally:
            self._connected = False

    def connect(self, timeout: float = 30.0) -> None:
        """Connect to the MCP server synchronously."""
        with self._lock:
            if self._connected:
                return
            loop = self._ensure_loop()
            started_future: concurrent.futures.Future[bool] = concurrent.futures.Future()

            def _schedule() -> None:
                self._worker_task = loop.create_task(self._run_worker(started_future))

            loop.call_soon_threadsafe(_schedule)
            started_future.result(timeout=timeout)

    async def connect_async(self, timeout: float = 30.0) -> None:
        """Connect to the MCP server asynchronously."""
        with self._lock:
            if self._connected:
                return
            loop = self._ensure_loop()
            started_future: concurrent.futures.Future[bool] = concurrent.futures.Future()

            def _schedule() -> None:
                self._worker_task = loop.create_task(self._run_worker(started_future))

            loop.call_soon_threadsafe(_schedule)
            await asyncio.wait_for(asyncio.wrap_future(started_future), timeout=timeout)

    def _send_cmd(self, action: str, *args: Any, timeout: float | None = 60.0, **kwargs: Any) -> Any:
        if not self._connected or self._cmd_queue is None or self._loop is None:
            self.connect()

        res_future: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def _put() -> None:
            if self._cmd_queue is not None:
                self._cmd_queue.put_nowait((action, args, kwargs, res_future))
            else:
                res_future.set_exception(RuntimeError("MCP session is not connected."))

        self._loop.call_soon_threadsafe(_put)
        return res_future.result(timeout=timeout)

    async def _send_cmd_async(self, action: str, *args: Any, timeout: float | None = 60.0, **kwargs: Any) -> Any:
        if not self._connected or self._cmd_queue is None or self._loop is None:
            await self.connect_async()

        res_future: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def _put() -> None:
            if self._cmd_queue is not None:
                self._cmd_queue.put_nowait((action, args, kwargs, res_future))
            else:
                res_future.set_exception(RuntimeError("MCP session is not connected."))

        self._loop.call_soon_threadsafe(_put)
        future_coro = asyncio.wrap_future(res_future)
        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(future_coro, timeout=timeout)
        return await future_coro

    def list_tools(self, timeout: float | None = 30.0) -> Any:
        """List raw MCP tools from the server synchronously."""
        return self._send_cmd("list_tools", timeout=timeout)

    async def list_tools_async(self, timeout: float | None = 30.0) -> Any:
        """List raw MCP tools from the server asynchronously."""
        return await self._send_cmd_async("list_tools", timeout=timeout)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = 60.0) -> str:
        """Execute a tool on the MCP server synchronously and return serialized text output."""
        ctx = create_child_trace_context()
        emit_event(
            MCPCallStartEvent(
                trace_context=ctx,
                server_name=self.config.name,
                tool_name=name,
                arguments=arguments or {},
            )
        )
        t0 = time.perf_counter()
        with trace_scope(ctx):
            try:
                result = self._send_cmd("call_tool", name, arguments=arguments or {}, timeout=timeout)
                duration_ms = (time.perf_counter() - t0) * 1000
                emit_event(
                    MCPCallEndEvent(
                        trace_context=ctx,
                        server_name=self.config.name,
                        tool_name=name,
                        status="ok",
                        duration_ms=duration_ms,
                    )
                )
                return _serialize_mcp_result(result)
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                err_msg = str(err)
                emit_event(
                    MCPCallEndEvent(
                        trace_context=ctx,
                        server_name=self.config.name,
                        tool_name=name,
                        status="error",
                        duration_ms=duration_ms,
                        error=err_msg,
                    )
                )
                emit_event(
                    ErrorEvent(
                        trace_context=ctx,
                        error_type=type(err).__name__,
                        message=err_msg,
                    )
                )
                raise

    async def call_tool_async(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = 60.0
    ) -> str:
        """Execute a tool on the MCP server asynchronously and return serialized text output."""
        ctx = create_child_trace_context()
        emit_event(
            MCPCallStartEvent(
                trace_context=ctx,
                server_name=self.config.name,
                tool_name=name,
                arguments=arguments or {},
            )
        )
        t0 = time.perf_counter()
        with trace_scope(ctx):
            try:
                result = await self._send_cmd_async("call_tool", name, arguments=arguments or {}, timeout=timeout)
                duration_ms = (time.perf_counter() - t0) * 1000
                emit_event(
                    MCPCallEndEvent(
                        trace_context=ctx,
                        server_name=self.config.name,
                        tool_name=name,
                        status="ok",
                        duration_ms=duration_ms,
                    )
                )
                return _serialize_mcp_result(result)
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                err_msg = str(err)
                emit_event(
                    MCPCallEndEvent(
                        trace_context=ctx,
                        server_name=self.config.name,
                        tool_name=name,
                        status="error",
                        duration_ms=duration_ms,
                        error=err_msg,
                    )
                )
                emit_event(
                    ErrorEvent(
                        trace_context=ctx,
                        error_type=type(err).__name__,
                        message=err_msg,
                    )
                )
                raise

    def list_resources(self, timeout: float | None = 30.0) -> Any:
        """List available resources on the MCP server synchronously."""
        return self._send_cmd("list_resources", timeout=timeout)

    async def list_resources_async(self, timeout: float | None = 30.0) -> Any:
        """List available resources on the MCP server asynchronously."""
        return await self._send_cmd_async("list_resources", timeout=timeout)

    def read_resource(self, uri: str, timeout: float | None = 30.0) -> Any:
        """Read a resource from the MCP server synchronously."""
        return self._send_cmd("read_resource", uri, timeout=timeout)

    async def read_resource_async(self, uri: str, timeout: float | None = 30.0) -> Any:
        """Read a resource from the MCP server asynchronously."""
        return await self._send_cmd_async("read_resource", uri, timeout=timeout)

    def list_prompts(self, timeout: float | None = 30.0) -> Any:
        """List available prompts on the MCP server synchronously."""
        return self._send_cmd("list_prompts", timeout=timeout)

    async def list_prompts_async(self, timeout: float | None = 30.0) -> Any:
        """List available prompts on the MCP server asynchronously."""
        return await self._send_cmd_async("list_prompts", timeout=timeout)

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = 30.0) -> Any:
        """Retrieve a rendered prompt from the MCP server synchronously."""
        return self._send_cmd("get_prompt", name, arguments=arguments or {}, timeout=timeout)

    async def get_prompt_async(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = 30.0
    ) -> Any:
        """Retrieve a rendered prompt from the MCP server asynchronously."""
        return await self._send_cmd_async("get_prompt", name, arguments=arguments or {}, timeout=timeout)

    async def get_evidor_tools(self, force_refresh: bool = False) -> tuple[Tool, ...]:
        """Convert all MCP tools exposed by this server into Evidor Tool objects."""
        if self._cached_tools is not None and not force_refresh:
            return self._cached_tools

        raw_tools_result = await self.list_tools_async()
        tools_list = getattr(raw_tools_result, "tools", raw_tools_result) or []
        evidor_tools: list[Tool] = []

        for mcp_tool in tools_list:
            evidor_tools.append(self._create_tool_adapter(mcp_tool))

        self._cached_tools = tuple(evidor_tools)
        return self._cached_tools

    def _create_tool_adapter(self, mcp_tool: Any) -> Tool:
        orig_name = mcp_tool.name
        if self.config.prefix:
            prefix = self.config.prefix
            tool_name = f"{prefix}_{orig_name}" if not prefix.endswith(("_", ":")) else f"{prefix}{orig_name}"
        else:
            tool_name = orig_name

        tool_desc = getattr(mcp_tool, "description", None) or f"MCP tool '{orig_name}' from server '{self.config.name}'."

        schema = getattr(mcp_tool, "input_schema", None) or getattr(mcp_tool, "inputSchema", None)
        if hasattr(schema, "model_dump"):
            params_schema = schema.model_dump()
        elif isinstance(schema, dict):
            params_schema = dict(schema)
        else:
            params_schema = {"type": "object", "properties": {}}

        if "type" not in params_schema:
            params_schema["type"] = "object"
        if "properties" not in params_schema:
            params_schema["properties"] = {}

        session = self

        # We define an async callable. Evidor Tool seamlessly executes async callables
        # both asynchronously (via await) and synchronously (via thread/coroutine runner).
        async def _mcp_callable(**kwargs: Any) -> str:
            return await session.call_tool_async(orig_name, kwargs)

        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=params_schema,
            func=_mcp_callable,
        )

    def close(self, timeout: float = 5.0) -> None:
        """Close the MCP session and terminate worker task."""
        with self._lock:
            if self._cmd_queue is not None and self._loop is not None:
                done_event = threading.Event()

                def _shutdown() -> None:
                    async def _do_stop() -> None:
                        if self._cmd_queue is not None:
                            self._cmd_queue.put_nowait(None)
                        if self._worker_task is not None:
                            try:
                                await self._worker_task
                            except Exception:
                                pass
                        done_event.set()

                    self._loop.create_task(_do_stop())

                self._loop.call_soon_threadsafe(_shutdown)
                done_event.wait(timeout=timeout)

            self._connected = False
            self._cmd_queue = None
            self._worker_task = None
            self._cached_tools = None

            if self._owns_loop and self._loop is not None:
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(timeout=timeout)
                self._loop = None
                self._thread = None

    async def close_async(self, timeout: float = 5.0) -> None:
        """Close the MCP session asynchronously."""
        done_future = concurrent.futures.Future()

        def _sync_close() -> None:
            try:
                self.close(timeout=timeout)
                done_future.set_result(True)
            except Exception as e:
                done_future.set_exception(e)

        threading.Thread(target=_sync_close, daemon=True).start()
        await asyncio.wrap_future(done_future)

    def __enter__(self) -> "MCPSession":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "MCPSession":
        await self.connect_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()

