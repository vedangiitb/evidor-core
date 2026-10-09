"""Tests for Arize Phoenix adapter with OpenInference semantics."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from evidor.agent import Agent
from evidor.models import GenerationRequest, GenerationResponse, ToolCall
from evidor.telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    PhoenixSink,
    TelemetryRuntime,
    TokenUsage,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TraceContext,
)
from evidor.tools import tool


class MockProvider:
    def __init__(self, responses: list[GenerationResponse]) -> None:
        self._responses = list(responses)
        self.model = "mock-gpt-4o"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._responses:
            raise RuntimeError("No more responses")
        return self._responses.pop(0)

    def with_model(self, model: str) -> "MockProvider":
        return self


def create_phoenix_test_tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_missing_phoenix_dependency_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.telemetry.adapters import phoenix as phoenix_module

    monkeypatch.setattr(phoenix_module, "HAS_PHOENIX", False)
    with pytest.raises(ImportError, match="pip install 'evidor\\[phoenix\\]'"):
        PhoenixSink()


def test_phoenix_openinference_attributes_enrichment() -> None:
    tp, exporter = create_phoenix_test_tracer()
    sink = PhoenixSink(tracer_provider=tp)

    root_ctx = TraceContext.new_root()
    llm_ctx = root_ctx.child()
    tool_ctx = llm_ctx.child()

    # 1. Start agent turn
    sink.write([
        AgentRunStartEvent(
            trace_context=root_ctx,
            run_id="run-phx-1",
            user_prompt="Calculate 2 + 2",
        )
    ])

    # 2. Start LLM call
    sink.write([
        LLMCallStartEvent(
            trace_context=llm_ctx,
            provider="openai",
            model="gpt-4o",
            input_prompt="Calculate 2 + 2",
        )
    ])

    # 3. Tool call execution
    sink.write([
        ToolCallStartEvent(
            trace_context=tool_ctx,
            tool_name="calculator",
            tool_call_id="call-c1",
            arguments={"expression": "2+2"},
        ),
        ToolCallEndEvent(
            trace_context=tool_ctx,
            tool_name="calculator",
            tool_call_id="call-c1",
            status="ok",
            duration_ms=15.0,
            result="4",
        ),
    ])

    # 4. LLM call end
    sink.write([
        LLMCallEndEvent(
            trace_context=llm_ctx,
            provider="openai",
            model="gpt-4o",
            status="ok",
            duration_ms=210.0,
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            output_text="The answer is 4.",
        )
    ])

    # 5. Agent run end
    sink.write([
        AgentRunEndEvent(
            trace_context=root_ctx,
            run_id="run-phx-1",
            status="ok",
            duration_ms=250.0,
            output_text="The answer is 4.",
        )
    ])

    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    # Map spans by name
    spans_by_name = {s.name: s for s in spans}
    tool_span = spans_by_name["tool.calculator"]
    llm_span = spans_by_name["llm.gpt-4o"]
    agent_span = spans_by_name["agent.run"]

    # Verify OpenInference attributes on tool span
    assert tool_span.attributes.get("openinference.span.kind") == "TOOL"
    assert tool_span.attributes.get("tool.name") == "calculator"
    assert tool_span.attributes.get("output.value") == "4"

    # Verify OpenInference attributes on LLM span
    assert llm_span.attributes.get("openinference.span.kind") == "LLM"
    assert llm_span.attributes.get("llm.model_name") == "gpt-4o"
    assert llm_span.attributes.get("input.value") == "Calculate 2 + 2"
    assert llm_span.attributes.get("output.value") == "The answer is 4."
    assert llm_span.attributes.get("llm.token_count.total") == 30

    # Verify OpenInference attributes on agent span
    assert agent_span.attributes.get("openinference.span.kind") == "AGENT"
    assert agent_span.attributes.get("input.value") == "Calculate 2 + 2"
    assert agent_span.attributes.get("output.value") == "The answer is 4."


def test_phoenix_end_to_end_agent_execution() -> None:
    tp, exporter = create_phoenix_test_tracer()
    sink = PhoenixSink(tracer_provider=tp)
    runtime = TelemetryRuntime(sinks=[sink], flush_interval_seconds=0.01)

    provider = MockProvider([
        GenerationResponse(
            text="Done with Phoenix!",
            model="mock-gpt-4o",
        )
    ])
    agent = Agent(provider=provider, telemetry=runtime)

    try:
        reply = agent.send("Hello Phoenix")
        assert reply.text == "Done with Phoenix!"
        runtime.flush()

        spans = exporter.get_finished_spans()
        assert len(spans) >= 2  # At least agent.run and llm span
        names = {s.name for s in spans}
        assert "agent.run" in names
        assert "llm.mock-gpt-4o" in names
    finally:
        runtime.close()
