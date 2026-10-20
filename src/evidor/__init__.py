"""Minimal, provider-agnostic LLM harness."""

from .agent import Agent
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
from .models import (
    AnyThinkingConfig,
    AnthropicThinkingConfig,
    AsyncStreamResponse,
    GeminiThinkingConfig,
    GenerationRequest,
    GenerationResponse,
    Message,
    OpenAIThinkingConfig,
    SearchResult,
    StreamChunk,
    StreamResponse,
    ThinkingConfig,
    ThinkingEffort,
    ToolCall,
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
    "AnthropicThinkingConfig",
    "AnyThinkingConfig",
    "AsyncStreamResponse",
    "BraveSearchProvider",
    "calculator",
    "clean_html",
    "ComparisonItem",
    "ConsoleSink",
    "EvidenceItem",
    "EvidenceStore",
    "evidence_store_tools",
    "ExaSearchProvider",
    "fetch_webpage_content",
    "GeminiProvider",
    "GeminiThinkingConfig",
    "GenerationRequest",
    "GenerationResponse",
    "get_current_time",
    "filesystem_tools",
    "HTTPServerConfig",
    "InMemorySink",
    "LangfuseSink",
    "MCPClient",
    "MCPServerConfig",
    "MCPSession",
    "Message",
    "ModelProvider",
    "OpenAIProvider",
    "OpenAIThinkingConfig",
    "OpenTelemetrySink",
    "PhoenixSink",
    "PrometheusSink",
    "read_webpage",
    "read_webpage_tool",
    "SearchResult",
    "SSEServerConfig",
    "StdioServerConfig",
    "StreamableHTTPServerConfig",
    "StreamChunk",
    "StreamResponse",
    "TelemetryRuntime",
    "TelemetrySink",
    "ThinkingConfig",
    "ThinkingEffort",
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

