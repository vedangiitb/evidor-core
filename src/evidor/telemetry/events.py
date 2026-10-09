"""Domain event dataclasses and trace context for Evidor telemetry.

These structures represent framework-neutral lifecycle events across agent runs,
LLM generations, tool executions, MCP invocations, retrieval operations, and errors.
"""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from . import semconv

SpanStatus = Literal["ok", "error", "cancelled"]


def _generate_trace_id() -> str:
    """Generate a standard 32-hex character trace ID (128-bit W3C compliant)."""
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    """Generate a standard 16-hex character span ID (64-bit W3C compliant)."""
    return secrets.token_hex(8)


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceContext:
    """Carries distributed tracing identity across agent lifecycle boundaries."""

    trace_id: str = field(default_factory=_generate_trace_id)
    span_id: str = field(default_factory=_generate_span_id)
    parent_span_id: str | None = None

    @classmethod
    def new_root(cls, trace_id: str | None = None) -> TraceContext:
        """Create a new root trace context."""
        return cls(
            trace_id=trace_id or _generate_trace_id(),
            span_id=_generate_span_id(),
            parent_span_id=None,
        )

    def child(self) -> TraceContext:
        """Derive a child trace context preserving the trace_id and linking the parent span."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=_generate_span_id(),
            parent_span_id=self.span_id,
        )

    def to_traceparent(self) -> str:
        """Format as a W3C traceparent header value (version 00, sampled)."""
        return f"00-{self.trace_id}-{self.span_id}-01"


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenUsage:
    """Normalized token counts for model generations."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(cls, prompt_tokens: int = 0, completion_tokens: int = 0) -> TokenUsage:
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TelemetryEvent:
    """Base framework-neutral telemetry event."""

    trace_context: TraceContext
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_attributes(self) -> dict[str, Any]:
        """Convert typed event fields into standard semantic attributes."""
        return dict(self.attributes)


# ==============================================================================
# Agent Run Lifecycle Events
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunStartEvent(TelemetryEvent):
    """Emitted when an agent execution turn begins."""

    run_id: str
    user_prompt: str = ""
    system_prompt: str | None = None

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_AGENT
        attrs[semconv.EVIDOR_AGENT_RUN_ID] = self.run_id
        if self.user_prompt:
            attrs["agent.prompt"] = self.user_prompt
        if self.system_prompt:
            attrs["agent.system_prompt"] = self.system_prompt
        return attrs


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunEndEvent(TelemetryEvent):
    """Emitted when an agent execution turn finishes."""

    run_id: str
    status: SpanStatus = "ok"
    duration_ms: float = 0.0
    iterations: int = 1
    output_text: str = ""
    error: str | None = None

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_AGENT
        attrs[semconv.EVIDOR_AGENT_RUN_ID] = self.run_id
        attrs[semconv.EVIDOR_AGENT_ITERATIONS] = self.iterations
        attrs["status"] = self.status
        attrs["duration_ms"] = self.duration_ms
        if self.output_text:
            attrs["agent.output"] = self.output_text
        if self.error:
            attrs[semconv.ERROR_MESSAGE] = self.error
        return attrs


# ==============================================================================
# LLM Generation Lifecycle Events
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCallStartEvent(TelemetryEvent):
    """Emitted when a call to an LLM provider starts."""

    provider: str
    model: str
    messages_count: int = 0
    tools_count: int = 0
    input_messages: tuple[dict[str, Any], ...] = ()
    input_prompt: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.input_messages, (list, tuple)):
            object.__setattr__(self, "input_messages", tuple(self.input_messages))

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_LLM
        attrs[semconv.GEN_AI_SYSTEM] = self.provider
        attrs[semconv.GEN_AI_REQUEST_MODEL] = self.model
        attrs["llm.messages_count"] = self.messages_count
        attrs["llm.tools_count"] = self.tools_count
        if self.input_prompt:
            attrs["llm.input_prompt"] = self.input_prompt
        if self.input_messages:
            attrs["llm.input_messages"] = list(self.input_messages)
        return attrs


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCallEndEvent(TelemetryEvent):
    """Emitted when a call to an LLM provider finishes."""

    provider: str
    model: str
    status: SpanStatus = "ok"
    duration_ms: float = 0.0
    token_usage: TokenUsage | None = None
    tool_calls_count: int = 0
    output_text: str = ""
    output_tool_calls: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.output_tool_calls, (list, tuple)):
            object.__setattr__(self, "output_tool_calls", tuple(self.output_tool_calls))

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_LLM
        attrs[semconv.GEN_AI_SYSTEM] = self.provider
        attrs[semconv.GEN_AI_RESPONSE_MODEL] = self.model
        attrs["status"] = self.status
        attrs["duration_ms"] = self.duration_ms
        attrs["llm.tool_calls_count"] = self.tool_calls_count
        if self.output_text:
            attrs["llm.output_text"] = self.output_text
        if self.output_tool_calls:
            attrs["llm.output_tool_calls"] = list(self.output_tool_calls)
        if self.token_usage:
            attrs[semconv.GEN_AI_USAGE_INPUT_TOKENS] = self.token_usage.prompt_tokens
            attrs[semconv.GEN_AI_USAGE_OUTPUT_TOKENS] = self.token_usage.completion_tokens
            attrs[semconv.GEN_AI_USAGE_TOTAL_TOKENS] = self.token_usage.total_tokens
        if self.error:
            attrs[semconv.ERROR_MESSAGE] = self.error
        return attrs


# ==============================================================================
# Tool Execution Lifecycle Events
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallStartEvent(TelemetryEvent):
    """Emitted when a local tool execution starts."""

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_TOOL
        attrs[semconv.EVIDOR_TOOL_NAME] = self.tool_name
        attrs[semconv.EVIDOR_TOOL_CALL_ID] = self.tool_call_id
        return attrs


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallEndEvent(TelemetryEvent):
    """Emitted when a local tool execution finishes."""

    tool_name: str
    tool_call_id: str
    status: SpanStatus = "ok"
    duration_ms: float = 0.0
    result: str | None = None
    error: str | None = None

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_TOOL
        attrs[semconv.EVIDOR_TOOL_NAME] = self.tool_name
        attrs[semconv.EVIDOR_TOOL_CALL_ID] = self.tool_call_id
        attrs["status"] = self.status
        attrs["duration_ms"] = self.duration_ms
        if self.result is not None:
            attrs["tool.output"] = self.result
        if self.error:
            attrs[semconv.ERROR_MESSAGE] = self.error
        return attrs


# ==============================================================================
# MCP Invocation Lifecycle Events
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPCallStartEvent(TelemetryEvent):
    """Emitted when an invocation to an external MCP server starts."""

    server_name: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_TOOL
        attrs[semconv.EVIDOR_MCP_SERVER_NAME] = self.server_name
        attrs[semconv.EVIDOR_TOOL_NAME] = self.tool_name
        return attrs


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPCallEndEvent(TelemetryEvent):
    """Emitted when an invocation to an external MCP server finishes."""

    server_name: str
    tool_name: str
    status: SpanStatus = "ok"
    duration_ms: float = 0.0
    error: str | None = None

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.OPENINFERENCE_SPAN_KIND] = semconv.SPAN_KIND_TOOL
        attrs[semconv.EVIDOR_MCP_SERVER_NAME] = self.server_name
        attrs[semconv.EVIDOR_TOOL_NAME] = self.tool_name
        attrs["status"] = self.status
        attrs["duration_ms"] = self.duration_ms
        if self.error:
            attrs[semconv.ERROR_MESSAGE] = self.error
        return attrs


# ==============================================================================
# Error and Retry Events
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(TelemetryEvent):
    """Emitted when an operational or model error occurs."""

    error_type: str
    message: str
    retry_count: int = 0
    outcome: str = "failed"
    details: dict[str, Any] = field(default_factory=dict)

    def to_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attributes)
        attrs[semconv.ERROR_TYPE] = self.error_type
        attrs[semconv.ERROR_MESSAGE] = self.message
        attrs[semconv.EVIDOR_RETRY_COUNT] = self.retry_count
        attrs[semconv.EVIDOR_RETRY_OUTCOME] = self.outcome
        attrs.update(self.details)
        return attrs


def redact_event(event: TelemetryEvent) -> TelemetryEvent:
    """Return a copy of the telemetry event with raw prompt, message, and tool contents stripped."""
    if isinstance(event, AgentRunStartEvent):
        return replace(event, user_prompt="", system_prompt=None)
    if isinstance(event, AgentRunEndEvent):
        return replace(event, output_text="")
    if isinstance(event, LLMCallStartEvent):
        return replace(event, input_prompt="", input_messages=())
    if isinstance(event, LLMCallEndEvent):
        return replace(event, output_text="", output_tool_calls=())
    if isinstance(event, ToolCallStartEvent):
        return replace(event, arguments={})
    if isinstance(event, ToolCallEndEvent):
        return replace(event, result=None)
    if isinstance(event, MCPCallStartEvent):
        return replace(event, arguments={})
    return event

