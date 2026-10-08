import asyncio
import json
from pathlib import Path
import sys
import pytest

from evidor import (
    Agent,
    GenerationRequest,
    GenerationResponse,
    MCPClient,
    MCPServerConfig,
    MCPSession,
    SSEServerConfig,
    StdioServerConfig,
    Tool,
    ToolCall,
)


AMAZON_MOCK_SERVER_SCRIPT = """
from mcp.server.mcpserver import MCPServer
server = MCPServer("amazon-mock")

@server.tool()
def search_products(query: str, max_results: int = 5) -> str:
    \"\"\"Search for products on Amazon.\"\"\"
    return f"Amazon results for '{query}': [Echo Dot, Kindle Paperwhite, Fire TV Stick]"

@server.tool()
def get_order_status(order_id: str) -> str:
    \"\"\"Get the shipping status of an Amazon order.\"\"\"
    if order_id == "error-id":
        raise ValueError("Invalid order ID provided.")
    return f"Order {order_id} is shipped and arriving tomorrow."

@server.resource("amazon://inventory/kindle")
def get_kindle_inventory() -> str:
    \"\"\"Check Kindle stock count.\"\"\"
    return "Stock: 42 units available in Seattle warehouse"

@server.prompt()
def deal_finder(category: str) -> str:
    \"\"\"Find daily deals in a category.\"\"\"
    return f"You are a deal finder. Suggest top discounts for {category}."

if __name__ == "__main__":
    server.run("stdio")
"""


GOOGLE_MOCK_SERVER_SCRIPT = """
from mcp.server.mcpserver import MCPServer
server = MCPServer("google-mock")

@server.tool()
def search_products(query: str) -> str:
    \"\"\"Google Shopping search.\"\"\"
    return f"Google Shopping results for '{query}': [Pixel 9, Nest Hub]"

if __name__ == "__main__":
    server.run("stdio")
"""


class FakeAgentProvider:
    """Mock LLM provider that simulates a 2-turn tool call conversation."""

    def __init__(self, tool_to_call: str, tool_args: dict, final_answer: str) -> None:
        self.tool_to_call = tool_to_call
        self.tool_args = tool_args
        self.final_answer = final_answer
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        if self.calls == 1:
            return GenerationResponse(
                text="",
                model="test-model",
                tool_calls=(
                    ToolCall(id="call_1", name=self.tool_to_call, arguments=self.tool_args),
                ),
            )
        # Second turn: receive tool output
        tool_msg = [m for m in request.messages if m.role == "tool"][0]
        return GenerationResponse(
            text=f"{self.final_answer}: {tool_msg.content}",
            model="test-model",
        )


mcp_available = False
try:
    import mcp  # noqa: F401
    mcp_available = True
except ImportError:
    pass

requires_mcp = pytest.mark.skipif(
    not mcp_available,
    reason="The 'mcp' package is required for live MCP tests. Install with: pip install 'evidor[mcp]'",
)

uvicorn_available = False
try:
    import uvicorn  # noqa: F401
    uvicorn_available = True
except ImportError:
    pass

requires_uvicorn = pytest.mark.skipif(
    not uvicorn_available,
    reason="The 'uvicorn' package is required for hosted MCP tests.",
)


def test_mcp_server_config_validation() -> None:
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        MCPServerConfig(name="")

    with pytest.raises(ValueError, match="requires a 'command'"):
        MCPServerConfig(name="test", transport="stdio", command=None)

    with pytest.raises(ValueError, match="requires a 'url'"):
        MCPServerConfig(name="test", transport="sse", url=None)

    with pytest.raises(ValueError, match="Unsupported transport 'grpc'"):
        MCPServerConfig(name="test", transport="grpc")  # type: ignore[arg-type]


def test_mcp_server_config_factories() -> None:
    stdio_cfg = StdioServerConfig("amazon", command="npx", args=["-y", "amazon-mcp"], prefix="amz")
    assert stdio_cfg.name == "amazon"
    assert stdio_cfg.command == "npx"
    assert stdio_cfg.args == ["-y", "amazon-mcp"]
    assert stdio_cfg.prefix == "amz"
    assert stdio_cfg.transport == "stdio"

    sse_cfg = SSEServerConfig("remote", url="http://localhost:8000/sse", headers={"Auth": "token"})
    assert sse_cfg.name == "remote"
    assert sse_cfg.url == "http://localhost:8000/sse"
    assert sse_cfg.headers == {"Auth": "token"}
    assert sse_cfg.transport == "sse"


def test_mcp_server_config_from_dict_and_file(tmp_path: Path) -> None:
    data = {
        "mcpServers": {
            "amazon": {
                "command": "uvx",
                "args": ["amazon-mcp-server"],
                "env": {"AWS_PROFILE": "default"},
            },
            "weather": {
                "url": "http://localhost:8080/sse",
            },
        }
    }
    config_file = tmp_path / "claude_config.json"
    config_file.write_text(json.dumps(data), encoding="utf-8")

    configs = MCPServerConfig.from_file(config_file)
    assert len(configs) == 2
    names = {c.name for c in configs}
    assert names == {"amazon", "weather"}

    amazon_cfg = next(c for c in configs if c.name == "amazon")
    assert amazon_cfg.transport == "stdio"
    assert amazon_cfg.command == "uvx"
    assert amazon_cfg.env == {"AWS_PROFILE": "default"}

    weather_cfg = next(c for c in configs if c.name == "weather")
    assert weather_cfg.transport == "sse"
    assert weather_cfg.url == "http://localhost:8080/sse"


def test_mcp_missing_dependency_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MCPSession raises a clear, actionable ImportError when 'mcp' is not installed."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.client", None)
    monkeypatch.setitem(sys.modules, "mcp.client.session", None)
    config = MCPServerConfig.stdio("mock", command="python")
    session = MCPSession(config)
    with pytest.raises(ImportError, match="The 'mcp' package is required for MCP support"):
        session.connect()


@pytest.mark.asyncio
async def test_mcp_missing_dependency_raises_import_error_async(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MCPSession raises a clear, actionable ImportError when 'mcp' is not installed (async)."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.client", None)
    monkeypatch.setitem(sys.modules, "mcp.client.session", None)
    config = MCPServerConfig.stdio("mock", command="python")
    session = MCPSession(config)
    with pytest.raises(ImportError, match="The 'mcp' package is required for MCP support"):
        await session.connect_async()


@requires_mcp
def test_mcp_session_stdio_lifecycle() -> None:
    config = MCPServerConfig.stdio(
        name="amazon",
        command=sys.executable,
        args=["-c", AMAZON_MOCK_SERVER_SCRIPT],
    )
    with MCPSession(config) as session:
        assert session.is_connected
        assert session.name == "amazon"

        # List tools
        tools_res = session.list_tools()
        tool_names = [t.name for t in tools_res.tools]
        assert "search_products" in tool_names
        assert "get_order_status" in tool_names

        # Call tool synchronously
        result = session.call_tool("search_products", {"query": "kindle"})
        assert "Amazon results for 'kindle'" in result
        assert "Kindle Paperwhite" in result

        # Call another tool
        order_res = session.call_tool("get_order_status", {"order_id": "ORD-99"})
        assert "Order ORD-99 is shipped" in order_res

        # Test prompt
        prompt = session.get_prompt("deal_finder", {"category": "electronics"})
        assert prompt.description == "Find daily deals in a category."

    assert not session.is_connected


@requires_mcp
@pytest.mark.asyncio
async def test_mcp_session_async_lifecycle() -> None:
    config = MCPServerConfig.stdio(
        name="amazon",
        command=sys.executable,
        args=["-c", AMAZON_MOCK_SERVER_SCRIPT],
    )
    async with MCPSession(config) as session:
        assert session.is_connected

        # List tools async
        tools_res = await session.list_tools_async()
        tool_names = [t.name for t in tools_res.tools]
        assert "search_products" in tool_names

        # Call tool async
        result = await session.call_tool_async("search_products", {"query": "tablet"})
        assert "Amazon results for 'tablet'" in result

        # Convert to Evidor tools and execute async
        evidor_tools = await session.get_evidor_tools()
        search_tool = next(t for t in evidor_tools if t.name == "search_products")
        async_res = await search_tool.execute_async(query="tablet")
        assert "Amazon results for 'tablet'" in async_res

    assert not session.is_connected


@requires_mcp
def test_mcp_client_multi_server_collision_handling() -> None:
    amazon_cfg = MCPServerConfig.stdio(
        name="amazon",
        command=sys.executable,
        args=["-c", AMAZON_MOCK_SERVER_SCRIPT],
    )
    google_cfg = MCPServerConfig.stdio(
        name="google",
        command=sys.executable,
        args=["-c", GOOGLE_MOCK_SERVER_SCRIPT],
    )

    with MCPClient(servers=[amazon_cfg, google_cfg]) as client:
        assert "amazon" in client.servers
        assert "google" in client.servers

        tools = client.get_tools()
        tool_names = [t.name for t in tools]

        # Both servers provide search_products -> should be auto-prefixed to prevent collision
        assert "amazon_search_products" in tool_names
        assert "google_search_products" in tool_names
        # Unique tool get_order_status remains unchanged
        assert "get_order_status" in tool_names

        # Execute both tools
        amz_tool = next(t for t in tools if t.name == "amazon_search_products")
        goog_tool = next(t for t in tools if t.name == "google_search_products")

        assert "Amazon results for 'shoes'" in amz_tool.execute(query="shoes")
        assert "Google Shopping results for 'shoes'" in goog_tool.execute(query="shoes")


@requires_mcp
def test_agent_sync_send_with_amazon_mcp() -> None:
    provider = FakeAgentProvider(
        tool_to_call="search_products",
        tool_args={"query": "noise cancelling headphones"},
        final_answer="Top recommendation",
    )

    with Agent(
        provider,
        mcp_servers={
            "amazon": {
                "command": sys.executable,
                "args": ["-c", AMAZON_MOCK_SERVER_SCRIPT],
            }
        },
    ) as agent:
        # Agent auto-discovers MCP tools
        tool_names = [t.name for t in agent.tools]
        assert "search_products" in tool_names
        assert "get_order_status" in tool_names

        response = agent.send("Find top headphones on Amazon")
        assert "Top recommendation" in response.text
        assert "Amazon results for 'noise cancelling headphones'" in response.text

        # Verify conversation history
        assert len(agent.messages) == 4
        assert agent.messages[0].role == "user"
        assert agent.messages[1].role == "assistant"
        assert agent.messages[2].role == "tool"
        assert "Amazon results for 'noise cancelling headphones'" in agent.messages[2].content
        assert agent.messages[3].role == "assistant"


@requires_mcp
@pytest.mark.asyncio
async def test_agent_async_send_with_amazon_mcp() -> None:
    provider = FakeAgentProvider(
        tool_to_call="get_order_status",
        tool_args={"order_id": "ORD-12345"},
        final_answer="Your order update",
    )

    async with Agent(
        provider,
        mcp_servers={
            "amazon": {
                "command": sys.executable,
                "args": ["-c", AMAZON_MOCK_SERVER_SCRIPT],
            }
        },
    ) as agent:
        response = await agent.send_async("What is the status of ORD-12345?")
        assert "Your order update" in response.text
        assert "Order ORD-12345 is shipped" in response.text


@requires_mcp
def test_agent_with_mcp_client_in_tools_list() -> None:
    client = MCPClient.from_stdio(
        command=sys.executable,
        args=["-c", AMAZON_MOCK_SERVER_SCRIPT],
        name="amazon",
    )

    provider = FakeAgentProvider(
        tool_to_call="search_products",
        tool_args={"query": "coffee maker"},
        final_answer="Found products",
    )

    with client:
        # Pass client directly in tools list
        agent = Agent(provider, tools=[client])
        assert any(t.name == "search_products" for t in agent.tools)

        response = agent.send("Search coffee maker")
        assert "Amazon results for 'coffee maker'" in response.text


@requires_mcp
def test_mcp_tool_error_handling() -> None:
    client = MCPClient.from_stdio(
        name="amazon",
        command=sys.executable,
        args=["-c", AMAZON_MOCK_SERVER_SCRIPT],
    )
    with client:
        tools = client.get_tools()
        order_tool = next(t for t in tools if t.name == "get_order_status")

        # Calling with invalid argument that raises an exception in the MCP server
        with pytest.raises(Exception) as exc_info:
            order_tool.execute(order_id="error-id")
        assert "get_order_status" in str(exc_info.value)


@requires_mcp
@requires_uvicorn
def test_remote_hosted_mcp_server_streamable_http() -> None:
    import socket
    import threading
    import time
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = MCPServer("streamable-aws-mock")

    @server.tool()
    def query_bedrock_model(model_id: str) -> str:
        """Query Amazon Bedrock foundation model."""
        return f"Bedrock model '{model_id}' status: ACTIVE and available"

    app = server.streamable_http_app()
    t = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="error"),
        daemon=True,
    )
    t.start()
    time.sleep(1.2)

    url = f"http://127.0.0.1:{port}/mcp"
    provider = FakeAgentProvider(
        tool_to_call="query_bedrock_model",
        tool_args={"model_id": "anthropic.claude-3-5-sonnet"},
        final_answer="Bedrock Query Result",
    )

    with Agent(
        provider,
        mcp_servers={
            "bedrock": {
                "url": url,
                "transport": "streamable-http",
                "headers": {"Authorization": "Bearer aws-session-token"},
            }
        },
    ) as agent:
        response = agent.send("Check Bedrock Claude status")
        assert "Bedrock Query Result" in response.text
        assert "anthropic.claude-3-5-sonnet' status: ACTIVE" in response.text


@requires_mcp
@requires_uvicorn
def test_remote_hosted_mcp_server_sse() -> None:
    import socket
    import threading
    import time
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    remote_server = MCPServer("hosted-aws-server")

    @remote_server.tool()
    def check_cloudwatch_alarm(alarm_name: str) -> str:
        """Check status of a CloudWatch alarm."""
        return f"Alarm '{alarm_name}' state: OK"

    app = remote_server.sse_app()
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1.2)

    url = f"http://127.0.0.1:{port}/sse"
    provider = FakeAgentProvider(
        tool_to_call="check_cloudwatch_alarm",
        tool_args={"alarm_name": "HighCPUUtilization"},
        final_answer="Monitoring Status",
    )

    with Agent(
        provider,
        mcp_servers={
            "cloudwatch": {
                "url": url,
                "headers": {"Authorization": "Bearer test-api-key"},
            }
        },
    ) as agent:
        response = agent.send("Check CPU alarm")
        assert "Monitoring Status" in response.text
        assert "Alarm 'HighCPUUtilization' state: OK" in response.text
