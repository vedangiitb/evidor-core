"""Tests for Langfuse adapter."""

from unittest.mock import MagicMock
import pytest

from evidor.agent import Agent
from evidor.models import GenerationRequest, GenerationResponse
from evidor.telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LangfuseSink,
    TelemetryRuntime,
    TokenUsage,
    ToolCallEndEvent,
    ToolCallStartEvent,
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


def test_missing_langfuse_dependency_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.telemetry.adapters import langfuse as lf_module

    monkeypatch.setattr(lf_module, "HAS_LANGFUSE", False)
    with pytest.raises(ImportError, match="pip install 'evidor\\[langfuse\\]'"):
        LangfuseSink()


def test_langfuse_observation_lifecycle() -> None:
    mock_client = MagicMock()
    mock_obs = MagicMock()
    mock_client.start_observation.return_value = mock_obs

    sink = LangfuseSink(client=mock_client)
    root_ctx = TraceContext.new_root()
    llm_ctx = root_ctx.child()
    tool_ctx = llm_ctx.child()

    # 1. Agent run start
    sink.write([
        AgentRunStartEvent(
            trace_context=root_ctx,
            run_id="run-lf-1",
            user_prompt="Summarize this text",
        )
    ])
    assert mock_client.start_observation.call_count == 1
    call1 = mock_client.start_observation.call_args[1]
    assert call1["name"] == "agent.run"
    assert call1["as_type"] == "agent"
    assert call1["input"] == "Summarize this text"
    assert call1["trace_context"]["trace_id"] == root_ctx.trace_id

    # 2. LLM call start
    sink.write([
        LLMCallStartEvent(
            trace_context=llm_ctx,
            provider="openai",
            model="gpt-4o",
            input_prompt="Summarize this text",
        )
    ])
    assert mock_client.start_observation.call_count == 2
    call2 = mock_client.start_observation.call_args[1]
    assert call2["name"] == "llm.gpt-4o"
    assert call2["as_type"] == "generation"
    assert call2["model"] == "gpt-4o"
    assert call2["trace_context"]["parent_span_id"] == root_ctx.span_id

    # 3. Tool call start & end
    sink.write([
        ToolCallStartEvent(
            trace_context=tool_ctx,
            tool_name="summarizer",
            tool_call_id="call-s1",
            arguments={"text": "hello"},
        ),
        ToolCallEndEvent(
            trace_context=tool_ctx,
            tool_name="summarizer",
            tool_call_id="call-s1",
            status="ok",
            duration_ms=50.0,
            result="summary text",
        ),
    ])
    assert mock_client.start_observation.call_count == 3
    call3 = mock_client.start_observation.call_args[1]
    assert call3["name"] == "tool.summarizer"
    assert call3["as_type"] == "tool"

    # 4. LLM call end
    sink.write([
        LLMCallEndEvent(
            trace_context=llm_ctx,
            provider="openai",
            model="gpt-4o",
            status="ok",
            duration_ms=180.0,
            token_usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40),
            output_text="Here is your summary.",
        )
    ])

    # 5. Agent run end
    sink.write([
        AgentRunEndEvent(
            trace_context=root_ctx,
            run_id="run-lf-1",
            status="ok",
            duration_ms=250.0,
            output_text="Here is your summary.",
        )
    ])

    # Ensure update and end calls were executed on observations
    assert mock_obs.update.call_count >= 3
    assert mock_obs.end.call_count >= 3


def test_langfuse_error_recording() -> None:
    mock_client = MagicMock()
    sink = LangfuseSink(client=mock_client)
    ctx = TraceContext.new_root()

    sink.write([
        ErrorEvent(
            trace_context=ctx,
            error_type="RateLimitError",
            message="Rate limit exceeded",
            details={"retry_after": 5},
        )
    ])

    assert mock_client.create_event.call_count == 1
    call = mock_client.create_event.call_args[1]
    assert call["name"] == "error"
    assert call["level"] == "ERROR"
    assert call["status_message"] == "Rate limit exceeded"
    assert call["metadata"]["error_type"] == "RateLimitError"
    assert call["metadata"]["retry_after"] == 5


def test_langfuse_flush_and_close() -> None:
    mock_client = MagicMock()
    sink = LangfuseSink(client=mock_client)

    sink.flush()
    assert mock_client.flush.call_count == 1

    sink.close()
    assert mock_client.shutdown.call_count == 1


def test_langfuse_end_to_end_agent() -> None:
    mock_client = MagicMock()
    mock_obs = MagicMock()
    mock_client.start_observation.return_value = mock_obs

    sink = LangfuseSink(client=mock_client)
    runtime = TelemetryRuntime(sinks=[sink], flush_interval_seconds=0.01)

    provider = MockProvider([
        GenerationResponse(
            text="Done via Langfuse!",
            model="mock-gpt-4o",
        )
    ])
    agent = Agent(provider=provider, telemetry=runtime)

    try:
        reply = agent.send("Hello Langfuse")
        assert reply.text == "Done via Langfuse!"
        runtime.flush()

        # Both agent.run and llm.mock-gpt-4o observations should have been created
        assert mock_client.start_observation.call_count >= 2
    finally:
        runtime.close()


def test_langfuse_capture_content_false() -> None:
    mock_client = MagicMock()
    mock_obs = MagicMock()
    mock_client.start_observation.return_value = mock_obs

    sink = LangfuseSink(client=mock_client, capture_content=False)
    ctx = TraceContext.new_root()

    sink.write([
        AgentRunStartEvent(
            trace_context=ctx,
            run_id="run-lf-redact",
            user_prompt="Confidential prompt",
        ),
        AgentRunEndEvent(
            trace_context=ctx,
            run_id="run-lf-redact",
            status="ok",
            output_text="Confidential response",
        ),
    ])

    # start_observation input must be None
    assert mock_client.start_observation.call_count == 1
    call_kwargs = mock_client.start_observation.call_args[1]
    assert call_kwargs["input"] is None
    assert call_kwargs["metadata"]["run_id"] == "run-lf-redact"

    # obs.update output must not be passed or None
    assert mock_obs.update.call_count == 1
    update_kwargs = mock_obs.update.call_args[1]
    assert "output" not in update_kwargs or update_kwargs["output"] is None


