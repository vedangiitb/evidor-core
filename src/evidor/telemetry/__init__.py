"""Evidor telemetry subsystem."""

from .context import (
    create_child_trace_context,
    emit_event,
    get_active_telemetry,
    get_current_trace_context,
    telemetry_scope,
    trace_scope,
)
from .events import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    MCPCallEndEvent,
    MCPCallStartEvent,
    SpanStatus,
    TelemetryEvent,
    TokenUsage,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TraceContext,
    redact_event,
)
from .runtime import OverflowStrategy, TelemetryRuntime
from .semconv import (
    EVIDOR_AGENT_ITERATIONS,
    EVIDOR_AGENT_RUN_ID,
    EVIDOR_MCP_SERVER_NAME,
    EVIDOR_RETRY_COUNT,
    EVIDOR_RETRY_DELAY_SECONDS,
    EVIDOR_RETRY_MAX,
    EVIDOR_RETRY_OUTCOME,
    EVIDOR_TOOL_CALL_ID,
    EVIDOR_TOOL_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    OPENINFERENCE_SPAN_KIND,
    SPAN_KIND_AGENT,
    SPAN_KIND_CHAIN,
    SPAN_KIND_LLM,
    SPAN_KIND_TOOL,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters import LangfuseSink, OpenTelemetrySink, PhoenixSink, PrometheusSink
from .sink import ConsoleSink, InMemorySink, TelemetrySink

_ADAPTER_EXPORTS = {
    "LangfuseSink": ".adapters.langfuse",
    "OpenTelemetrySink": ".adapters.opentelemetry",
    "PhoenixSink": ".adapters.phoenix",
    "PrometheusSink": ".adapters.prometheus",
}

__all__ = [
    # Trace Context and Base Types
    "TraceContext",
    "TokenUsage",
    "SpanStatus",
    "TelemetryEvent",
    "redact_event",
    # Context & Scoping
    "trace_scope",
    "telemetry_scope",
    "get_current_trace_context",
    "get_active_telemetry",
    "create_child_trace_context",
    "emit_event",
    # Runtime & Sinks
    "TelemetryRuntime",
    "OverflowStrategy",
    "TelemetrySink",
    "InMemorySink",
    "ConsoleSink",
    "OpenTelemetrySink",
    "LangfuseSink",
    "PhoenixSink",
    "PrometheusSink",
    # Agent Run
    "AgentRunStartEvent",
    "AgentRunEndEvent",
    # LLM Call
    "LLMCallStartEvent",
    "LLMCallEndEvent",
    # Tool Call
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    # MCP Call
    "MCPCallStartEvent",
    "MCPCallEndEvent",
    # Errors & Retries
    "ErrorEvent",
    # Semantic Conventions
    "GEN_AI_SYSTEM",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_TOTAL_TOKENS",
    "OPENINFERENCE_SPAN_KIND",
    "SPAN_KIND_AGENT",
    "SPAN_KIND_CHAIN",
    "SPAN_KIND_LLM",
    "SPAN_KIND_TOOL",
    "EVIDOR_AGENT_RUN_ID",
    "EVIDOR_AGENT_ITERATIONS",
    "EVIDOR_TOOL_NAME",
    "EVIDOR_TOOL_CALL_ID",
    "EVIDOR_MCP_SERVER_NAME",
    "EVIDOR_RETRY_COUNT",
    "EVIDOR_RETRY_DELAY_SECONDS",
    "EVIDOR_RETRY_MAX",
    "EVIDOR_RETRY_OUTCOME",
]


def __getattr__(name: str) -> Any:
    if name in _ADAPTER_EXPORTS:
        import importlib

        module = importlib.import_module(_ADAPTER_EXPORTS[name], __package__)
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


