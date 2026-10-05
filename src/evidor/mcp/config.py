"""Configuration models and loaders for Model Context Protocol (MCP) servers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

TransportType = Literal["stdio", "sse", "streamable-http", "http"]


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server over stdio, Streamable HTTP, or SSE."""

    name: str
    transport: TransportType = "stdio"
    # stdio parameters
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    encoding: str = "utf-8"
    # remote http / sse parameters
    url: str | None = None
    headers: dict[str, Any] | None = None
    timeout: float = 30.0
    sse_read_timeout: float = 300.0
    # tool naming
    prefix: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MCPServerConfig 'name' must be a non-empty string.")
        # Normalize "http" alias to "streamable-http"
        if self.transport == "http":
            self.transport = "streamable-http"

        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"MCPServerConfig '{self.name}' with transport='stdio' requires a 'command'.")
        elif self.transport in ("sse", "streamable-http"):
            if not self.url:
                raise ValueError(f"MCPServerConfig '{self.name}' with transport='{self.transport}' requires a 'url'.")
        else:
            raise ValueError(
                f"Unsupported transport '{self.transport}'. Expected 'stdio', 'streamable-http', 'http', or 'sse'."
            )

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: list[str] | Sequence[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        encoding: str = "utf-8",
        prefix: str | None = None,
    ) -> "MCPServerConfig":
        """Create a stdio-based MCP server configuration."""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=list(args) if args is not None else [],
            env=env,
            cwd=cwd,
            encoding=encoding,
            prefix=prefix,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: dict[str, Any] | None = None,
        timeout: float = 30.0,
        prefix: str | None = None,
    ) -> "MCPServerConfig":
        """Create a modern Streamable HTTP MCP server configuration."""
        return cls(
            name=name,
            transport="streamable-http",
            url=url,
            headers=headers,
            timeout=timeout,
            prefix=prefix,
        )

    @classmethod
    def streamable_http(
        cls,
        name: str,
        url: str,
        *,
        headers: dict[str, Any] | None = None,
        timeout: float = 30.0,
        prefix: str | None = None,
    ) -> "MCPServerConfig":
        """Alias for http() to explicitly designate Streamable HTTP transport."""
        return cls.http(name=name, url=url, headers=headers, timeout=timeout, prefix=prefix)

    @classmethod
    def sse(
        cls,
        name: str,
        url: str,
        *,
        headers: dict[str, Any] | None = None,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        prefix: str | None = None,
    ) -> "MCPServerConfig":
        """Create a legacy SSE/HTTP-based MCP server configuration."""
        return cls(
            name=name,
            transport="sse",
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            prefix=prefix,
        )

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "MCPServerConfig":
        """Parse an MCP server configuration from a dictionary (e.g. Claude Desktop config)."""
        transport = data.get("transport")
        if transport is None:
            if "url" in data:
                url_str = str(data["url"]).lower()
                # If explicit SSE path, default to sse, otherwise modern streamable-http
                transport = "sse" if "/sse" in url_str else "streamable-http"
            else:
                transport = "stdio"

        if transport in ("streamable-http", "http"):
            return cls.http(
                name=name,
                url=data["url"],
                headers=data.get("headers"),
                timeout=float(data.get("timeout", 30.0)),
                prefix=data.get("prefix"),
            )

        if transport == "sse":
            return cls.sse(
                name=name,
                url=data["url"],
                headers=data.get("headers"),
                timeout=float(data.get("timeout", 30.0)),
                sse_read_timeout=float(data.get("sse_read_timeout", 300.0)),
                prefix=data.get("prefix"),
            )

        return cls.stdio(
            name=name,
            command=data["command"],
            args=data.get("args", []),
            env=data.get("env"),
            cwd=data.get("cwd"),
            encoding=data.get("encoding", "utf-8"),
            prefix=data.get("prefix"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> list["MCPServerConfig"]:
        """Load MCP server configurations from a JSON file (e.g. claude_desktop_config.json)."""
        file_path = Path(path).resolve(strict=True)
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)

        servers_dict = data.get("mcpServers", data)
        if not isinstance(servers_dict, dict):
            raise ValueError(f"Expected a dictionary of servers in {path}, got {type(servers_dict).__name__}")

        configs: list[MCPServerConfig] = []
        for server_name, server_data in servers_dict.items():
            if isinstance(server_data, dict):
                configs.append(cls.from_dict(server_name, server_data))
        return configs

    @classmethod
    def normalize_many(cls, configs: Any) -> list["MCPServerConfig"]:
        """Normalize various input formats (dict, list, path, single config) into a list of MCPServerConfig."""
        if configs is None:
            return []
        if isinstance(configs, MCPServerConfig):
            return [configs]
        if isinstance(configs, (str, Path)):
            return cls.from_file(configs)
        if isinstance(configs, dict):
            servers = configs.get("mcpServers", configs)
            result: list[MCPServerConfig] = []
            for name, cfg in servers.items():
                if isinstance(cfg, MCPServerConfig):
                    result.append(cfg)
                elif isinstance(cfg, dict):
                    result.append(cls.from_dict(name, cfg))
                else:
                    raise TypeError(
                        f"Invalid server config for '{name}': expected dict or MCPServerConfig, got {type(cfg).__name__}"
                    )
            return result
        if isinstance(configs, (list, tuple)):
            result = []
            for item in configs:
                if isinstance(item, MCPServerConfig):
                    result.append(item)
                elif isinstance(item, dict):
                    name = item.get("name", f"server_{len(result)}")
                    result.append(cls.from_dict(name, item))
                else:
                    raise TypeError(f"Expected MCPServerConfig or dict in list, got {type(item).__name__}")
            return result
        raise TypeError(f"Cannot parse MCP servers from {type(configs).__name__}")


# Aliases for convenience
StdioServerConfig = MCPServerConfig.stdio
HTTPServerConfig = MCPServerConfig.http
StreamableHTTPServerConfig = MCPServerConfig.streamable_http
SSEServerConfig = MCPServerConfig.sse
