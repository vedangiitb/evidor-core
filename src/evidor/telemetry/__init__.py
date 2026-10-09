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
)
from .runtime import OverflowStrategy, TelemetryRuntime
from .semconv import (
    EVIDOR_AGENT_ITERATIONS,
    EVIDOR_AGENT_RUN_ID,
    EVIDOR_MCP_SERVER_NAME,
    EVIDOR_RETRY_COUNT,
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
from .adapters import LangfuseSink, OpenTelemetrySink, PhoenixSink, PrometheusSink
from .sink import ConsoleSink, InMemorySink, TelemetrySink

__all__ = [
    # Trace Context and Base Types
    "TraceContext",
    "TokenUsage",
    "SpanStatus",
    "TelemetryEvent",
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
    "EVIDOR_RETRY_OUTCOME",
]
