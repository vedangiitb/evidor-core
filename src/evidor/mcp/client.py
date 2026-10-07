"""High-level multi-server MCP client for Evidor."""

import asyncio
from collections.abc import Sequence
import concurrent.futures
from pathlib import Path
import threading
from typing import Any

from evidor.tools import Tool

from .config import MCPServerConfig
from .session import MCPSession


class MCPClient:
    """Manages connections to one or more MCP servers and exposes unified Evidor tools."""

    def __init__(
        self,
        servers: Any = None,
        *,
        prefix_tools: bool = False,
    ) -> None:
        self._prefix_tools = prefix_tools
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._lock = threading.Lock()
        self._connected = False

        configs = MCPServerConfig.normalize_many(servers) if servers is not None else []
        self._configs: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, MCPSession] = {}

        for cfg in configs:
            self._add_config(cfg)

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
            target=_run, daemon=True, name="Evidor-MCP-Client-Loop"
        )
        self._thread.start()
        self._ready_event.wait()
        return self._loop

    def _add_config(self, cfg: MCPServerConfig) -> None:
        if self._prefix_tools and not cfg.prefix:
            cfg = MCPServerConfig(
                name=cfg.name,
                transport=cfg.transport,
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
                cwd=cfg.cwd,
                encoding=cfg.encoding,
                url=cfg.url,
                headers=cfg.headers,
                timeout=cfg.timeout,
                sse_read_timeout=cfg.sse_read_timeout,
                prefix=cfg.name,
            )
        self._configs[cfg.name] = cfg

    def add_server(
        self,
        server: MCPServerConfig | dict[str, Any],
        name: str | None = None,
    ) -> "MCPClient":
        """Add an MCP server configuration to this client."""
        with self._lock:
            if isinstance(server, dict):
                server_name = name or server.get("name", f"server_{len(self._configs)}")
                cfg = MCPServerConfig.from_dict(server_name, server)
            elif isinstance(server, MCPServerConfig):
                cfg = server
            else:
                raise TypeError(f"Expected MCPServerConfig or dict, got {type(server).__name__}")

            self._add_config(cfg)
            if self._connected:
                loop = self._ensure_loop()
                session = MCPSession(cfg, loop=loop)
                session.connect()
                self._sessions[cfg.name] = session
        return self

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | Sequence[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        name: str = "default",
        prefix: str | None = None,
    ) -> "MCPClient":
        """Create an MCPClient connected to a single stdio server."""
        cfg = MCPServerConfig.stdio(name=name, command=command, args=args, env=env, cwd=cwd, prefix=prefix)
        return cls(servers=[cfg])

    @classmethod
    def from_sse(
        cls,
        url: str,
        *,
        headers: dict[str, Any] | None = None,
        name: str = "default",
        prefix: str | None = None,
    ) -> "MCPClient":
        """Create an MCPClient connected to a single SSE server."""
        cfg = MCPServerConfig.sse(name=name, url=url, headers=headers, prefix=prefix)
        return cls(servers=[cfg])

    @classmethod
    def from_http(
        cls,
        url: str,
        *,
        headers: dict[str, Any] | None = None,
        name: str = "default",
        prefix: str | None = None,
    ) -> "MCPClient":
        """Create an MCPClient connected to a modern Streamable HTTP server."""
        cfg = MCPServerConfig.http(name=name, url=url, headers=headers, prefix=prefix)
        return cls(servers=[cfg])

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        transport: str | None = None,
        headers: dict[str, Any] | None = None,
        name: str = "default",
        prefix: str | None = None,
    ) -> "MCPClient":
        """Create an MCPClient from a remote URL, auto-detecting transport if omitted."""
        data: dict[str, Any] = {"url": url, "headers": headers, "prefix": prefix}
        if transport:
            data["transport"] = transport
        cfg = MCPServerConfig.from_dict(name, data)
        return cls(servers=[cfg])

    @classmethod
    def from_file(cls, path: str | Path, *, prefix_tools: bool = False) -> "MCPClient":
        """Create an MCPClient from a config file (e.g. claude_desktop_config.json)."""
        configs = MCPServerConfig.from_file(path)
        return cls(servers=configs, prefix_tools=prefix_tools)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, prefix_tools: bool = False) -> "MCPClient":
        """Create an MCPClient from a dictionary of server configurations."""
        configs = MCPServerConfig.normalize_many(data)
        return cls(servers=configs, prefix_tools=prefix_tools)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def servers(self) -> tuple[str, ...]:
        """Names of configured servers."""
        return tuple(self._configs.keys())

    def session(self, name: str) -> MCPSession:
        """Get the MCPSession for a specific server."""
        with self._lock:
            if name not in self._configs:
                raise KeyError(f"Server '{name}' is not configured on this MCPClient.")
            if name not in self._sessions:
                loop = self._ensure_loop()
                self._sessions[name] = MCPSession(self._configs[name], loop=loop)
            return self._sessions[name]

    def __getitem__(self, name: str) -> MCPSession:
        return self.session(name)

    def connect(self, timeout: float = 60.0) -> None:
        """Connect to all configured MCP servers in parallel synchronously."""
        with self._lock:
            if self._connected:
                return
            loop = self._ensure_loop()
            for name, cfg in self._configs.items():
                if name not in self._sessions:
                    self._sessions[name] = MCPSession(cfg, loop=loop)

            async def _connect_all() -> None:
                tasks = [session.connect_async(timeout=timeout) for session in self._sessions.values()]
                if tasks:
                    await asyncio.gather(*tasks)

            future = asyncio.run_coroutine_threadsafe(_connect_all(), loop)
            future.result(timeout=timeout)
            self._connected = True

    async def connect_async(self, timeout: float = 60.0) -> None:
        """Connect to all configured MCP servers in parallel asynchronously."""
        with self._lock:
            if self._connected:
                return
            loop = self._ensure_loop()
            for name, cfg in self._configs.items():
                if name not in self._sessions:
                    self._sessions[name] = MCPSession(cfg, loop=loop)

        # Connect sessions in parallel
        tasks = [session.connect_async(timeout=timeout) for session in self._sessions.values()]
        if tasks:
            await asyncio.gather(*tasks)
        self._connected = True

    def get_tools(self, force_refresh: bool = False) -> tuple[Tool, ...]:
        """Retrieve all Evidor Tool objects from all connected MCP servers in parallel."""
        if not self._connected:
            self.connect()

        tools: list[Tool] = []
        tool_counts: dict[str, int] = {}

        # Fetch tools from all sessions in parallel
        async def _fetch_all_tools() -> list[tuple[Tool, ...]]:
            tasks = [session.get_evidor_tools(force_refresh=force_refresh) for session in self._sessions.values()]
            return await asyncio.gather(*tasks) if tasks else []

        future = asyncio.run_coroutine_threadsafe(_fetch_all_tools(), self._ensure_loop())
        results = future.result(timeout=60.0)
        all_server_tools = list(zip(self._sessions.keys(), results))

        for _, st in all_server_tools:
            for t in st:
                tool_counts[t.name] = tool_counts.get(t.name, 0) + 1

        for server_name, s_tools in all_server_tools:
            for t in s_tools:
                # If collision detected and not already prefixed, prefix with server name
                if tool_counts[t.name] > 1 and not t.name.startswith(f"{server_name}_"):
                    prefixed_name = f"{server_name}_{t.name}"
                    t = Tool(
                        name=prefixed_name,
                        description=t.description,
                        parameters=t.parameters,
                        func=t.func,
                        timeout=t.timeout,
                    )
                tools.append(t)

        return tuple(tools)

    async def get_tools_async(self, force_refresh: bool = False) -> tuple[Tool, ...]:
        """Asynchronously retrieve all Evidor Tool objects from all connected MCP servers."""
        if not self._connected:
            await self.connect_async()

        tools: list[Tool] = []
        tool_counts: dict[str, int] = {}

        tasks = [session.get_evidor_tools(force_refresh=force_refresh) for session in self._sessions.values()]
        results = await asyncio.gather(*tasks) if tasks else []

        all_server_tools = list(zip(self._sessions.keys(), results))
        for _, st in all_server_tools:
            for t in st:
                tool_counts[t.name] = tool_counts.get(t.name, 0) + 1

        for server_name, s_tools in all_server_tools:
            for t in s_tools:
                if tool_counts[t.name] > 1 and not t.name.startswith(f"{server_name}_"):
                    prefixed_name = f"{server_name}_{t.name}"
                    t = Tool(
                        name=prefixed_name,
                        description=t.description,
                        parameters=t.parameters,
                        func=t.func,
                        timeout=t.timeout,
                    )
                tools.append(t)

        return tuple(tools)

    def close(self, timeout: float = 5.0) -> None:
        """Close all active MCP server sessions and shut down the client loop."""
        with self._lock:
            for session in self._sessions.values():
                try:
                    session.close(timeout=timeout)
                except Exception:
                    pass
            self._sessions.clear()
            self._connected = False

            if self._loop is not None:
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(timeout=timeout)
                self._loop = None
                self._thread = None

    async def close_async(self, timeout: float = 5.0) -> None:
        """Close all active MCP server sessions asynchronously."""
        done_future = concurrent.futures.Future()

        def _sync_close() -> None:
            try:
                self.close(timeout=timeout)
                done_future.set_result(True)
            except Exception as e:
                done_future.set_exception(e)

        threading.Thread(target=_sync_close, daemon=True).start()
        await asyncio.wrap_future(done_future)

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "MCPClient":
        await self.connect_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

