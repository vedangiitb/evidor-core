"""Centralized error message constants and formatting helpers for Evidor."""

# -----------------------------------------------------------------------------
# Generic event loop / synchronous call restriction messages
# -----------------------------------------------------------------------------

DEFAULT_SYNC_ERROR = (
    "This sync method cannot be called from a running event loop. "
    "Use its async equivalent instead."
)


def sync_forbidden_error(action: str, alternative: str | None = None) -> str:
    """Generate a standard error message for forbidden synchronous invocation in an event loop."""
    if alternative:
        return f"{action} cannot be called from a running event loop. Use '{alternative}' instead."
    return f"{action} cannot be called from a running event loop."


# -----------------------------------------------------------------------------
# Model Provider error constants
# -----------------------------------------------------------------------------

def format_provider_generate_error(provider_name: str) -> str:
    """Format the sync error message for a model provider's generate() method."""
    return (
        f"{provider_name}.generate() cannot be called from a running event loop. "
        "Use 'await provider.generate_async()' instead."
    )


OPENAI_GENERATE_SYNC_ERROR = format_provider_generate_error("OpenAIProvider")
ANTHROPIC_GENERATE_SYNC_ERROR = format_provider_generate_error("AnthropicProvider")
GEMINI_GENERATE_SYNC_ERROR = format_provider_generate_error("GeminiProvider")


# -----------------------------------------------------------------------------
# Agent error constants
# -----------------------------------------------------------------------------

AGENT_SEND_SYNC_ERROR = (
    "Agent.send() cannot be called from a running event loop. "
    "Use 'await agent.send_async()' instead."
)

AGENT_STREAM_SYNC_ERROR = (
    "Agent.send(..., stream=True) cannot be called from a running event loop. "
    "Use 'await agent.send_async(..., stream=True)' instead."
)

AGENT_CLOSE_SYNC_ERROR = (
    "Agent.close() cannot be called from a running event loop. "
    "Use 'await agent.close_async()' or 'async with agent:' instead."
)

AGENT_SYNC_MCP_TOOLS_ERROR = (
    "Cannot synchronize MCP tools synchronously from a running event loop. "
    "Access agent.tools outside an event loop or run an async agent turn first."
)


# -----------------------------------------------------------------------------
# MCP error constants
# -----------------------------------------------------------------------------

MCP_CLIENT_CONNECT_ERROR = (
    "MCPClient.connect() cannot be called from a running event loop. "
    "Use 'await client.connect_async()' instead."
)

MCP_CLIENT_GET_TOOLS_ERROR = (
    "MCPClient.get_tools() cannot be called from a running event loop. "
    "Use 'await client.get_tools_async()' instead."
)

MCP_CLIENT_CLOSE_ERROR = (
    "MCPClient.close() cannot be called from a running event loop. "
    "Use 'await client.close_async()' or 'async with client:' instead."
)

MCP_SESSION_CONNECT_ERROR = (
    "MCPSession.connect() cannot be called from a running event loop. "
    "Use 'await session.connect_async()' instead."
)

MCP_SESSION_LIST_TOOLS_ERROR = (
    "MCPSession.list_tools() cannot be called from a running event loop. "
    "Use 'await session.list_tools_async()' instead."
)

MCP_SESSION_CALL_TOOL_ERROR = (
    "MCPSession.call_tool() cannot be called from a running event loop. "
    "Use 'await session.call_tool_async()' instead."
)

MCP_SESSION_CLOSE_ERROR = (
    "MCPSession.close() cannot be called from a running event loop. "
    "Use 'await session.close_async()' or 'async with session:' instead."
)


# -----------------------------------------------------------------------------
# Tools & Executor error constants
# -----------------------------------------------------------------------------

TOOL_EXECUTOR_EXECUTE_ERROR = (
    "ToolExecutor.execute() cannot be called from a running event loop. "
    "Use 'await executor.execute_async()' instead."
)

ASYNC_TOOL_EXECUTE_ERROR = (
    "Async tool cannot be executed synchronously from a running event loop. "
    "Use 'await tool.execute_async()' instead."
)

