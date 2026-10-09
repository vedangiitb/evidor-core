"""Tests for OpenTelemetry adapter and OpenInference semantic compatibility."""

from typing import Any
import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from evidor.agent import Agent
from evidor.models import GenerationRequest, GenerationResponse, ToolCall
from evidor.telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    OpenTelemetrySink,
    TokenUsage,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TraceContext,
)
from evidor.telemetry.semconv import (
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_TOTAL_TOKENS,
    OPENINFERENCE_SPAN_KIND,
    SPAN_KIND_AGENT,
    SPAN_KIND_LLM,
    SPAN_KIND_TOOL,
)


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


def create_test_tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_missing_opentelemetry_dependency_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.telemetry.adapters import opentelemetry as otel_module

    monkeypatch.setattr(otel_module, "HAS_OPENTELEMETRY", False)
    with pytest.raises(ImportError, match="pip install 'evidor\\[otel\\]'"):
        otel_module.OpenTelemetrySink()


def test_opentelemetry_agent_run_and_llm_spans() -> None:
    tracer_provider, exporter = create_test_tracer()
    sink = OpenTelemetrySink(tracer_provider=tracer_provider)

    provider = MockProvider([
        GenerationResponse(text="Paris is the capital of France.", model="mock-gpt-4o"),
    ])

    agent = Agent(provider=provider, telemetry=sink)
    response = agent.send("What is the capital of France?")
    assert response.text == "Paris is the capital of France."

    agent.close()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Spans are completed in inside-out order: LLM span ends first, then Agent span
    llm_span = next(s for s in spans if s.name.startswith("llm"))
    agent_span = next(s for s in spans if s.name == "agent.run")

    # Trace hierarchy: LLM span's parent must be the Agent span!
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == agent_span.context.span_id
    assert llm_span.context.trace_id == agent_span.context.trace_id

    # OpenInference conventions
    assert agent_span.attributes[OPENINFERENCE_SPAN_KIND] == SPAN_KIND_AGENT
    assert agent_span.attributes["agent.prompt"] == "What is the capital of France?"
    assert agent_span.attributes["agent.output"] == "Paris is the capital of France."
    assert agent_span.status.status_code == StatusCode.OK

    assert llm_span.attributes[OPENINFERENCE_SPAN_KIND] == SPAN_KIND_LLM
    assert llm_span.attributes[GEN_AI_SYSTEM] == "MockProvider"
    assert llm_span.status.status_code == StatusCode.OK


def test_opentelemetry_tool_span_hierarchy() -> None:
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    tracer_provider, exporter = create_test_tracer()
    sink = OpenTelemetrySink(tracer_provider=tracer_provider)

    provider = MockProvider([
        GenerationResponse(
            text="",
            model="mock-gpt-4o",
            tool_calls=(ToolCall(id="call_99", name="multiply", arguments={"a": 6, "b": 7}),),
        ),
        GenerationResponse(text="The result is 42", model="mock-gpt-4o"),
    ])

    agent = Agent(provider=provider, tools=[multiply], telemetry=sink)
    response = agent.send("Multiply 6 and 7")
    assert response.text == "The result is 42"

    agent.close()

    spans = exporter.get_finished_spans()
    # 2 LLM calls + 1 Tool call + 1 Agent run = 4 spans
    assert len(spans) == 4

    agent_span = next(s for s in spans if s.name == "agent.run")
    tool_span = next(s for s in spans if s.name == "tool.multiply")

    # Tool is child of Agent run
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert tool_span.context.trace_id == agent_span.context.trace_id

    # Tool attributes
    assert tool_span.attributes[OPENINFERENCE_SPAN_KIND] == SPAN_KIND_TOOL
    assert tool_span.attributes["evidor.tool.name"] == "multiply"
    assert tool_span.attributes["tool.output"] == "42"
    assert tool_span.status.status_code == StatusCode.OK


def test_opentelemetry_error_recording() -> None:
    tracer_provider, exporter = create_test_tracer()
    sink = OpenTelemetrySink(tracer_provider=tracer_provider)

    class FailingProvider:
        model = "failing-model"

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise RuntimeError("API timeout")

        def with_model(self, model: str) -> "FailingProvider":
            return self

    agent = Agent(provider=FailingProvider(), telemetry=sink)

    with pytest.raises(RuntimeError, match="API timeout"):
        agent.send("Trigger failure")

    agent.close()

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = next(s for s in spans if s.name == "agent.run")
    assert agent_span.status.status_code == StatusCode.ERROR
    assert "API timeout" in (agent_span.status.description or "")


def test_opentelemetry_orphan_end_event_fallback() -> None:
    tracer_provider, exporter = create_test_tracer()
    sink = OpenTelemetrySink(tracer_provider=tracer_provider)

    ctx = TraceContext.new_root()
    # Write only an EndEvent without preceding StartEvent
    end_event = ToolCallEndEvent(
        trace_context=ctx,
        tool_name="standalone_tool",
        tool_call_id="call_orphan",
        status="ok",
        duration_ms=25.0,
        result="Success",
    )

    sink.write([end_event])
    sink.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "tool.standalone_tool"
    assert span.attributes["tool.output"] == "Success"
    assert span.status.status_code == StatusCode.OK

