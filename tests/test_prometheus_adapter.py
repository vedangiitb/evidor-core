"""Tests for Prometheus metrics adapter."""

import pytest
from prometheus_client import CollectorRegistry

from evidor.agent import Agent
from evidor.models import GenerationRequest, GenerationResponse
from evidor.telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    MCPCallEndEvent,
    PrometheusSink,
    TelemetryRuntime,
    TokenUsage,
    ToolCallEndEvent,
    TraceContext,
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


def test_missing_prometheus_dependency_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.telemetry.adapters import prometheus as prom_module

    monkeypatch.setattr(prom_module, "HAS_PROMETHEUS", False)
    with pytest.raises(ImportError, match="pip install 'evidor\\[prometheus\\]'"):
        PrometheusSink()


def test_prometheus_agent_run_metrics() -> None:
    registry = CollectorRegistry()
    sink = PrometheusSink(registry=registry)
    ctx = TraceContext()

    # Start agent run -> gauge incremented
    start_event = AgentRunStartEvent(
        trace_context=ctx,
        run_id="run-1",
        attributes={"agent_name": "researcher"},
    )
    sink.write([start_event])

    active = registry.get_sample_value("evidor_active_agent_runs", {"agent_name": "researcher"})
    assert active == 1.0

    # End agent run -> gauge decremented, total counter incremented
    end_event = AgentRunEndEvent(
        trace_context=ctx,
        run_id="run-1",
        status="ok",
        duration_ms=450.0,
        attributes={"agent_name": "researcher"},
    )
    sink.write([end_event])

    active = registry.get_sample_value("evidor_active_agent_runs", {"agent_name": "researcher"})
    assert active == 0.0

    completed = registry.get_sample_value("evidor_agent_runs_total", {"agent_name": "researcher", "status": "ok"})
    assert completed == 1.0


def test_prometheus_llm_and_token_metrics() -> None:
    registry = CollectorRegistry()
    sink = PrometheusSink(registry=registry)
    ctx = TraceContext()

    event = LLMCallEndEvent(
        trace_context=ctx,
        provider="openai",
        model="gpt-4o",
        status="ok",
        duration_ms=250.0,
        token_usage=TokenUsage(prompt_tokens=15, completion_tokens=35, total_tokens=50),
    )
    sink.write([event])

    llm_calls = registry.get_sample_value(
        "evidor_llm_calls_total",
        {"provider": "openai", "model": "gpt-4o", "status": "ok"},
    )
    assert llm_calls == 1.0

    prompt_toks = registry.get_sample_value(
        "evidor_tokens_total",
        {"provider": "openai", "model": "gpt-4o", "token_type": "prompt"},
    )
    assert prompt_toks == 15.0

    comp_toks = registry.get_sample_value(
        "evidor_tokens_total",
        {"provider": "openai", "model": "gpt-4o", "token_type": "completion"},
    )
    assert comp_toks == 35.0

    total_toks = registry.get_sample_value(
        "evidor_tokens_total",
        {"provider": "openai", "model": "gpt-4o", "token_type": "total"},
    )
    assert total_toks == 50.0


def test_prometheus_tool_and_mcp_and_error_metrics() -> None:
    registry = CollectorRegistry()
    sink = PrometheusSink(registry=registry)
    ctx = TraceContext()

    tool_end = ToolCallEndEvent(
        trace_context=ctx,
        tool_name="calculator",
        tool_call_id="call-1",
        status="ok",
        duration_ms=12.0,
    )
    mcp_end = MCPCallEndEvent(
        trace_context=ctx,
        server_name="github",
        tool_name="create_issue",
        status="ok",
        duration_ms=88.0,
    )
    err = ErrorEvent(
        trace_context=ctx,
        error_type="NetworkError",
        message="connection timed out",
    )

    sink.write([tool_end, mcp_end, err])

    assert registry.get_sample_value("evidor_tool_calls_total", {"tool_name": "calculator", "status": "ok"}) == 1.0
    assert (
        registry.get_sample_value(
            "evidor_mcp_calls_total",
            {"server_name": "github", "tool_name": "create_issue", "status": "ok"},
        )
        == 1.0
    )
    assert registry.get_sample_value("evidor_errors_total", {"error_type": "NetworkError"}) == 1.0


def test_prometheus_export_text() -> None:
    registry = CollectorRegistry()
    sink = PrometheusSink(registry=registry)
    ctx = TraceContext()
    sink.write([ErrorEvent(trace_context=ctx, error_type="AuthError", message="forbidden")])

    text = sink.export_text()
    assert "evidor_errors_total" in text
    assert 'error_type="AuthError"' in text


def test_prometheus_end_to_end_agent_execution() -> None:
    registry = CollectorRegistry()
    sink = PrometheusSink(registry=registry)
    runtime = TelemetryRuntime(sinks=[sink], flush_interval_seconds=0.01)

    provider = MockProvider([
        GenerationResponse(
            text="Done!",
            model="mock-gpt-4o",
        )
    ])
    agent = Agent(provider=provider, telemetry=runtime)

    try:
        reply = agent.send("Hello agent")
        assert reply.text == "Done!"
        runtime.flush()

        completed = registry.get_sample_value("evidor_agent_runs_total", {"agent_name": "default", "status": "ok"})
        assert completed == 1.0

        llm_count = registry.get_sample_value(
            "evidor_llm_calls_total",
            {"provider": "MockProvider", "model": "mock-gpt-4o", "status": "ok"},
        )
        assert llm_count == 1.0
    finally:
        runtime.close()
