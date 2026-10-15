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


def test_phoenix_falls_back_when_register_uses_removed_otlp_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Newer OTEL exporters must not prevent Phoenix telemetry from starting."""
    from evidor.telemetry.adapters import phoenix as phoenix_module

    def incompatible_register(**_kwargs: object) -> None:
        raise AttributeError("'HTTPSpanExporter' object has no attribute '_headers'")

    monkeypatch.setattr(phoenix_module.potel, "register", incompatible_register)

    with pytest.warns(RuntimeWarning, match="using direct OTLP HTTP export"):
        sink = PhoenixSink(
            endpoint="http://localhost:6006/v1/traces",
            project_name="fallback-project",
        )

    assert sink._tracer_provider.resource.attributes["openinference.project.name"] == "fallback-project"


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


def test_phoenix_capture_content_false() -> None:
    tp, exporter = create_phoenix_test_tracer()
    sink = PhoenixSink(tracer_provider=tp, capture_content=False)

    ctx = TraceContext.new_root()
    sink.write([
        AgentRunStartEvent(
            trace_context=ctx,
            run_id="run-phx-redact",
            user_prompt="Secret prompt",
        ),
        AgentRunEndEvent(
            trace_context=ctx,
            run_id="run-phx-redact",
            status="ok",
            output_text="Secret response",
        ),
    ])

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify input.value and output.value are NOT present
    assert "input.value" not in span.attributes
    assert "output.value" not in span.attributes

    # Verify OpenInference metadata IS present
    assert span.attributes.get("openinference.span.kind") == "AGENT"
    assert span.attributes.get("evidor.agent.run_id") == "run-phx-redact"


def test_phoenix_tool_call_response_enrichment() -> None:
    """Verify PhoenixSink properly populates output.value and OpenInference output messages when LLM generates tool calls."""
    tp, exporter = create_phoenix_test_tracer()
    sink = PhoenixSink(tracer_provider=tp)

    root_ctx = TraceContext.new_root()
    llm_ctx = root_ctx.child()

    sink.write([
        LLMCallStartEvent(
            trace_context=llm_ctx,
            provider="gemini",
            model="gemini-2.5-flash",
            input_prompt="Research python agent frameworks",
            input_messages=({"role": "user", "content": "Research python agent frameworks"},),
        ),
        LLMCallEndEvent(
            trace_context=llm_ctx,
            provider="gemini",
            model="gemini-2.5-flash",
            status="ok",
            duration_ms=180.0,
            token_usage=TokenUsage(prompt_tokens=150, completion_tokens=35, total_tokens=185),
            output_text="",
            output_tool_calls=({
                "id": "call_1",
                "name": "web_search",
                "arguments": {"query": "python agent frameworks comparison"},
            },),
        ),
    ])

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    llm_span = spans[0]

    # Verify input messages
    assert llm_span.attributes.get("input.value") == "Research python agent frameworks"
    assert llm_span.attributes.get("llm.input_messages.0.message.role") == "user"
    assert llm_span.attributes.get("llm.input_messages.0.message.content") == "Research python agent frameworks"

    # Verify output.value is populated with tool calls instead of being empty or fallback model JSON
    output_val = llm_span.attributes.get("output.value")
    assert output_val is not None
    assert "web_search" in output_val
    assert "python agent frameworks comparison" in output_val

    # Verify OpenInference structured tool call output message attributes
    assert llm_span.attributes.get("llm.output_messages.0.message.role") == "assistant"
    assert llm_span.attributes.get("llm.output_messages.0.message.tool_calls.0.tool_call.id") == "call_1"
    assert llm_span.attributes.get("llm.output_messages.0.message.tool_calls.0.tool_call.function.name") == "web_search"
    assert "python agent frameworks comparison" in llm_span.attributes.get("llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")

    # Verify token count metrics
    assert llm_span.attributes.get("llm.token_count.prompt") == 150
    assert llm_span.attributes.get("llm.token_count.completion") == 35
    assert llm_span.attributes.get("llm.token_count.total") == 185


def test_phoenix_reasoning_structured_output_attributes() -> None:
    tp, exporter = create_phoenix_test_tracer()
    sink = PhoenixSink(tracer_provider=tp, capture_content=True, capture_thinking=True)

    ctx = TraceContext.new_root()
    sink.write([
        LLMCallStartEvent(
            trace_context=ctx,
            provider="anthropic",
            model="claude-3-7-sonnet-20250219",
            input_prompt="Solve riddle",
        ),
        LLMCallEndEvent(
            trace_context=ctx,
            provider="anthropic",
            model="claude-3-7-sonnet-20250219",
            duration_ms=450.0,
            token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=250,
                total_tokens=350,
                reasoning_tokens=180,
            ),
            output_text="The answer is 42",
            thinking_text="Step 1: Deliberate\nStep 2: Conclude",
        ),
    ])

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify structured message contents without scalar collision
    attrs = span.attributes
    assert attrs.get("llm.output_messages.0.message.role") == "assistant"
    assert attrs.get("llm.output_messages.0.message.contents.0.message_content.type") == "reasoning"
    assert attrs.get("llm.output_messages.0.message.contents.0.message_content.text") == "Step 1: Deliberate\nStep 2: Conclude"
    assert attrs.get("llm.output_messages.0.message.contents.1.message_content.type") == "text"
    assert attrs.get("llm.output_messages.0.message.contents.1.message_content.text") == "The answer is 42"
    assert "llm.output_messages.0.message.content" not in attrs

    # Top-level output.value is final visible text
    assert attrs.get("output.value") == "The answer is 42"

    # Token count reasoning breakdown
    assert attrs.get("llm.token_count.completion_details.reasoning") == 180



