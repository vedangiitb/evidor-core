"""Tests for telemetry event dataclasses and trace context."""

import pytest

from evidor.telemetry.events import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    MCPCallEndEvent,
    MCPCallStartEvent,
    TelemetryEvent,
    TokenUsage,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TraceContext,
)
from evidor.telemetry.semconv import (
    EVIDOR_AGENT_RUN_ID,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_TOTAL_TOKENS,
    OPENINFERENCE_SPAN_KIND,
    SPAN_KIND_AGENT,
    SPAN_KIND_LLM,
)


def test_trace_context_hierarchy() -> None:
    root = TraceContext.new_root()
    assert len(root.trace_id) == 32
    assert len(root.span_id) == 16
    assert root.parent_span_id is None
    assert root.to_traceparent() == f"00-{root.trace_id}-{root.span_id}-01"

    child = root.child()
    assert child.trace_id == root.trace_id
    assert child.parent_span_id == root.span_id
    assert child.span_id != root.span_id
    assert child.to_traceparent() == f"00-{child.trace_id}-{child.span_id}-01"

    grandchild = child.child()
    assert grandchild.trace_id == root.trace_id
    assert grandchild.parent_span_id == child.span_id


def test_token_usage() -> None:
    usage = TokenUsage.from_counts(prompt_tokens=100, completion_tokens=50)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150

    # Immutability
    with pytest.raises(AttributeError):
        usage.prompt_tokens = 200  # type: ignore[misc]


def test_agent_run_events() -> None:
    ctx = TraceContext.new_root()
    start = AgentRunStartEvent(trace_context=ctx, run_id="run-123", user_prompt="Hello agent")
    assert start.run_id == "run-123"
    assert start.user_prompt == "Hello agent"
    start_attrs = start.to_attributes()
    assert start_attrs[OPENINFERENCE_SPAN_KIND] == SPAN_KIND_AGENT
    assert start_attrs[EVIDOR_AGENT_RUN_ID] == "run-123"
    assert start_attrs["agent.prompt"] == "Hello agent"

    end = AgentRunEndEvent(
        trace_context=ctx,
        run_id="run-123",
        status="ok",
        duration_ms=45.2,
        iterations=2,
        output_text="Agent answer",
    )
    assert end.duration_ms == 45.2
    end_attrs = end.to_attributes()
    assert end_attrs["status"] == "ok"
    assert end_attrs["agent.output"] == "Agent answer"


def test_llm_call_events() -> None:
    ctx = TraceContext.new_root().child()
    start = LLMCallStartEvent(
        trace_context=ctx,
        provider="openai",
        model="gpt-4o",
        messages_count=3,
        input_prompt="What is the weather?",
        input_messages=[{"role": "user", "content": "What is the weather?"}],
    )
    start_attrs = start.to_attributes()
    assert start_attrs[OPENINFERENCE_SPAN_KIND] == SPAN_KIND_LLM
    assert start_attrs[GEN_AI_SYSTEM] == "openai"
    assert start_attrs["llm.input_prompt"] == "What is the weather?"
    assert start_attrs["llm.input_messages"] == [{"role": "user", "content": "What is the weather?"}]

    usage = TokenUsage.from_counts(40, 60)
    end = LLMCallEndEvent(
        trace_context=ctx,
        provider="openai",
        model="gpt-4o",
        status="ok",
        duration_ms=500.0,
        token_usage=usage,
        output_text="The weather is sunny.",
        output_tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
    )
    end_attrs = end.to_attributes()
    assert end_attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 100
    assert end_attrs["duration_ms"] == 500.0
    assert end_attrs["llm.output_text"] == "The weather is sunny."
    assert end_attrs["llm.output_tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_tool_call_events() -> None:
    ctx = TraceContext.new_root().child()
    start = ToolCallStartEvent(
        trace_context=ctx,
        tool_name="calculator",
        tool_call_id="call_1",
        arguments={"expression": "2+2"},
    )
    assert start.tool_name == "calculator"
    assert start.arguments == {"expression": "2+2"}

    end = ToolCallEndEvent(
        trace_context=ctx,
        tool_name="calculator",
        tool_call_id="call_1",
        status="ok",
        duration_ms=1.5,
        result="4",
    )
    assert end.result == "4"


def test_mcp_call_events() -> None:
    ctx = TraceContext.new_root().child()
    start = MCPCallStartEvent(
        trace_context=ctx,
        server_name="filesystem",
        tool_name="read_file",
        arguments={"path": "/tmp/test.txt"},
    )
    assert start.server_name == "filesystem"

    end = MCPCallEndEvent(
        trace_context=ctx,
        server_name="filesystem",
        tool_name="read_file",
        status="ok",
        duration_ms=12.0,
    )
    assert end.duration_ms == 12.0


def test_error_event() -> None:
    ctx = TraceContext.new_root()
    err = ErrorEvent(
        trace_context=ctx,
        error_type="RateLimitError",
        message="Too many requests",
        retry_count=2,
        outcome="retrying",
    )
    attrs = err.to_attributes()
    assert attrs["error.type"] == "RateLimitError"
    assert attrs["evidor.retry.count"] == 2
    assert attrs["evidor.retry.outcome"] == "retrying"


def test_event_immutability() -> None:
    ctx = TraceContext.new_root()
    event = AgentRunStartEvent(trace_context=ctx, run_id="r1")
    with pytest.raises(AttributeError):
        event.run_id = "r2"  # type: ignore[misc]
