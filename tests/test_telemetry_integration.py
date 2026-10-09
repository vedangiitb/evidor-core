"""Tests for telemetry integration, trace propagation, and actor runtime."""

import time
from typing import Any

import pytest

from evidor.agent import Agent
from evidor.models import GenerationRequest, GenerationResponse, ToolCall
from evidor.telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    InMemorySink,
    LLMCallEndEvent,
    LLMCallStartEvent,
    TelemetryRuntime,
    TokenUsage,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TraceContext,
    emit_event,
)


class MockProvider:
    def __init__(self, responses: list[GenerationResponse]) -> None:
        self._responses = list(responses)
        self.model = "mock-model"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._responses:
            raise RuntimeError("No more mocked responses available.")
        return self._responses.pop(0)

    def with_model(self, model: str) -> "MockProvider":
        return self


def test_agent_telemetry_sync_flow() -> None:
    sink = InMemorySink()
    provider = MockProvider([
        GenerationResponse(text="Hello world!", model="mock-model"),
    ])

    agent = Agent(provider=provider, telemetry=sink)
    response = agent.send("Hi")
    assert response.text == "Hello world!"

    agent.close()

    events = sink.events
    assert len(events) == 4

    # 1. AgentRunStartEvent
    start_agent = events[0]
    assert isinstance(start_agent, AgentRunStartEvent)
    assert start_agent.user_prompt == "Hi"
    root_trace_id = start_agent.trace_context.trace_id
    root_span_id = start_agent.trace_context.span_id
    assert start_agent.trace_context.parent_span_id is None

    # 2. LLMCallStartEvent
    start_llm = events[1]
    assert isinstance(start_llm, LLMCallStartEvent)
    assert start_llm.trace_context.trace_id == root_trace_id
    assert start_llm.trace_context.parent_span_id == root_span_id
    assert start_llm.model == "mock-model"
    assert start_llm.input_prompt == "Hi"

    # 3. LLMCallEndEvent
    end_llm = events[2]
    assert isinstance(end_llm, LLMCallEndEvent)
    assert end_llm.trace_context.trace_id == root_trace_id
    assert end_llm.trace_context.span_id == start_llm.trace_context.span_id
    assert end_llm.output_text == "Hello world!"
    assert end_llm.status == "ok"
    assert end_llm.duration_ms >= 0

    # 4. AgentRunEndEvent
    end_agent = events[3]
    assert isinstance(end_agent, AgentRunEndEvent)
    assert end_agent.trace_context.trace_id == root_trace_id
    assert end_agent.trace_context.span_id == root_span_id
    assert end_agent.output_text == "Hello world!"
    assert end_agent.status == "ok"
    assert end_agent.duration_ms >= 0


def test_agent_telemetry_with_tool_call() -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    sink = InMemorySink()
    provider = MockProvider([
        GenerationResponse(
            text="",
            model="mock-model",
            tool_calls=(ToolCall(id="call_1", name="add", arguments={"a": 3, "b": 7}),),
        ),
        GenerationResponse(text="The sum is 10", model="mock-model"),
    ])

    agent = Agent(provider=provider, tools=[add], telemetry=sink)
    response = agent.send("Add 3 and 7")
    assert response.text == "The sum is 10"

    agent.close()

    events = sink.events
    assert len(events) == 8

    # Hierarchy verification
    agent_start = sink.filter(AgentRunStartEvent)[0]
    root_span_id = agent_start.trace_context.span_id

    tool_start = sink.filter(ToolCallStartEvent)[0]
    assert tool_start.tool_name == "add"
    assert tool_start.trace_context.parent_span_id == root_span_id

    tool_end = sink.filter(ToolCallEndEvent)[0]
    assert tool_end.result == "10"
    assert tool_end.status == "ok"
    assert tool_end.trace_context.span_id == tool_start.trace_context.span_id

    agent_end = sink.filter(AgentRunEndEvent)[0]
    assert agent_end.output_text == "The sum is 10"
    assert agent_end.iterations == 2


@pytest.mark.asyncio
async def test_agent_telemetry_async_flow() -> None:
    sink = InMemorySink()
    provider = MockProvider([
        GenerationResponse(text="Async result", model="mock-model"),
    ])

    agent = Agent(provider=provider, telemetry=sink)
    response = await agent.send_async("Async test")
    assert response.text == "Async result"

    agent.close()

    events = sink.events
    assert len(events) == 4
    assert isinstance(events[0], AgentRunStartEvent)
    assert isinstance(events[1], LLMCallStartEvent)
    assert isinstance(events[2], LLMCallEndEvent)
    assert isinstance(events[3], AgentRunEndEvent)


def test_agent_telemetry_error_handling() -> None:
    class FailingProvider:
        model = "failing-model"

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise ValueError("Provider connection failed")

        def with_model(self, model: str) -> "FailingProvider":
            return self

    sink = InMemorySink()
    agent = Agent(provider=FailingProvider(), telemetry=sink)

    with pytest.raises(ValueError, match="Provider connection failed"):
        agent.send("Fail please")

    agent.close()

    agent_ends = sink.filter(AgentRunEndEvent)
    assert len(agent_ends) == 1
    assert agent_ends[0].status == "error"
    assert "Provider connection failed" in (agent_ends[0].error or "")

    errors = sink.filter(ErrorEvent)
    assert len(errors) >= 1
    assert errors[0].error_type == "ValueError"
    assert "Provider connection failed" in errors[0].message


def test_runtime_bounded_queue_drop_newest() -> None:
    sink = InMemorySink()
    runtime = TelemetryRuntime(
        [sink],
        queue_size=3,
        batch_size=10,
        flush_interval_seconds=10.0,  # don't drain immediately
        overflow_strategy="drop_newest",
    )

    ctx = TraceContext.new_root()
    # Fill queue to capacity (3 items)
    for i in range(5):
        runtime.emit(AgentRunStartEvent(trace_context=ctx, run_id=f"r_{i}"))

    assert runtime.dropped_count == 2
    runtime.flush(timeout=0.5)
    runtime.close()

    recorded = sink.events
    assert len(recorded) == 3
    # First 3 were kept, newest 2 were dropped
    assert [e.run_id for e in recorded] == ["r_0", "r_1", "r_2"]  # type: ignore[attr-defined]


def test_runtime_bounded_queue_drop_oldest() -> None:
    sink = InMemorySink()
    runtime = TelemetryRuntime(
        [sink],
        queue_size=3,
        batch_size=10,
        flush_interval_seconds=10.0,
        overflow_strategy="drop_oldest",
    )

    ctx = TraceContext.new_root()
    for i in range(5):
        runtime.emit(AgentRunStartEvent(trace_context=ctx, run_id=f"r_{i}"))

    assert runtime.dropped_count == 2
    runtime.flush(timeout=0.5)
    runtime.close()

    recorded = sink.events
    assert len(recorded) == 3
    # Oldest 2 were dropped, newest 3 kept
    assert [e.run_id for e in recorded] == ["r_2", "r_3", "r_4"]  # type: ignore[attr-defined]


def test_mcp_call_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.mcp.config import MCPServerConfig
    from evidor.mcp.session import MCPSession
    from evidor.telemetry import MCPCallEndEvent, MCPCallStartEvent, telemetry_scope

    config = MCPServerConfig(name="test_server", transport="stdio", command="python")
    session = MCPSession(config)
    sink = InMemorySink()
    runtime = TelemetryRuntime([sink])

    def mock_send_cmd(self: Any, cmd: str, *args: Any, **kwargs: Any) -> Any:
        return "mcp tool success"

    monkeypatch.setattr(MCPSession, "_send_cmd", mock_send_cmd)

    with telemetry_scope(runtime):
        res = session.call_tool("echo", {"msg": "hi"})
        assert res == "mcp tool success"

    runtime.flush()
    runtime.close()

    mcp_starts = sink.filter(MCPCallStartEvent)
    mcp_ends = sink.filter(MCPCallEndEvent)
    assert len(mcp_starts) == 1
    assert mcp_starts[0].server_name == "test_server"
    assert mcp_starts[0].tool_name == "echo"
    assert mcp_starts[0].arguments == {"msg": "hi"}

    assert len(mcp_ends) == 1
    assert mcp_ends[0].server_name == "test_server"
    assert mcp_ends[0].status == "ok"
    assert mcp_ends[0].duration_ms >= 0
