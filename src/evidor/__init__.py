"""Minimal, provider-agnostic LLM harness."""

from .agent import Agent
from .agent.context.context import (
    DEFAULT_CHAIN_MILESTONES,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_HIGH_WATER_RATIO,
    DEFAULT_LOW_WATER_RATIO,
    DEFAULT_MAX_MESSAGES,
)
from .agent.providers import AnthropicProvider, GeminiProvider, ModelProvider, OpenAIProvider
from .mcp import (
    HTTPServerConfig,
    MCPClient,
    MCPServerConfig,
    MCPSession,
    SSEServerConfig,
    StdioServerConfig,
    StreamableHTTPServerConfig,
)
from .models import GenerationRequest, GenerationResponse, Message, SearchResult, ToolCall
from .retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    calculate_backoff_delay,
    extract_retry_after,
    is_network_or_provider_error,
)


from typing import TYPE_CHECKING, Any

from .telemetry import (
    ConsoleSink,
    InMemorySink,
    TelemetryRuntime,
    TelemetrySink,
    TraceContext,
)

if TYPE_CHECKING:
    from .telemetry import (
        LangfuseSink,
        OpenTelemetrySink,
        PhoenixSink,
        PrometheusSink,
    )

_LAZY_EXPORTS = {
    "LangfuseSink": ".telemetry",
    "OpenTelemetrySink": ".telemetry",
    "PhoenixSink": ".telemetry",
    "PrometheusSink": ".telemetry",
}
from .tools import (
    BraveSearchProvider,
    calculator,
    clean_html,
    ComparisonItem,
    EvidenceItem,
    EvidenceStore,
    evidence_store_tools,
    ExaSearchProvider,
    fetch_webpage_content,
    filesystem_tools,
    get_current_time,
    read_webpage,
    read_webpage_tool,
    TavilySearchProvider,
    Tool,
    tool,
    web_search,
    WebSearchProvider,
)

__all__ = [
    "Agent",
    "AnthropicProvider",
    "BraveSearchProvider",
    "calculator",
    "calculate_backoff_delay",
    "clean_html",
    "ComparisonItem",
    "ConsoleSink",
    "DEFAULT_CHAIN_MILESTONES",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_HIGH_WATER_RATIO",
    "DEFAULT_LOW_WATER_RATIO",
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_RETRY_CONFIG",
    "EvidenceItem",
    "EvidenceStore",
    "evidence_store_tools",
    "ExaSearchProvider",
    "extract_retry_after",
    "fetch_webpage_content",
    "GeminiProvider",
    "GenerationRequest",
    "GenerationResponse",
    "get_current_time",
    "filesystem_tools",
    "HTTPServerConfig",
    "InMemorySink",
    "is_network_or_provider_error",
    "LangfuseSink",
    "MCPClient",
    "MCPServerConfig",
    "MCPSession",
    "Message",
    "ModelProvider",
    "OpenAIProvider",
    "OpenTelemetrySink",
    "PhoenixSink",
    "PrometheusSink",
    "read_webpage",
    "read_webpage_tool",
    "RetryConfig",
    "SearchResult",
    "SSEServerConfig",
    "StdioServerConfig",
    "StreamableHTTPServerConfig",
    "TelemetryRuntime",
    "TelemetrySink",
    "Tool",
    "ToolCall",
    "TraceContext",
    "TavilySearchProvider",
    "tool",
    "WebSearchProvider",
    "web_search",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name], __package__)
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))

