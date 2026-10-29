import asyncio
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evidor import (
    Agent,
    AnthropicProvider,
    AsyncStreamResponse,
    GeminiProvider,
    GenerationRequest,
    GenerationResponse,
    InMemorySink,
    Message,
    OpenAIProvider,
    StreamChunk,
    StreamResponse,
    TelemetryRuntime,
    Tool,
    ToolCall,
)
from evidor.agent.providers.base import ToolCallAccumulator
from evidor.telemetry.events import LLMCallEndEvent, LLMCallStartEvent


# ==============================================================================
# Model & Accumulator Tests
# ==============================================================================


def test_stream_chunk_stringification_and_defaults() -> None:
    chunk = StreamChunk(text="Hello world")
    assert str(chunk) == "Hello world"
    assert chunk.text == "Hello world"
    assert chunk.thinking is None
    assert chunk.tool_calls == ()
    assert chunk.is_finished is False
    assert chunk.response is None

    # Test tuple conversion
    tc = ToolCall(id="c1", name="tool", arguments={"a": 1})
    chunk_with_tools = StreamChunk(text="", tool_calls=[tc])
    assert isinstance(chunk_with_tools.tool_calls, tuple)
    assert chunk_with_tools.tool_calls[0].name == "tool"


def test_tool_call_accumulator_multi_chunk_assembly() -> None:
    acc = ToolCallAccumulator()
    acc.start_call("call_123", "search")

    # Arguments arrive split across 3 chunks
    acc.add_arguments_delta("call_123", '{"que')
    acc.add_arguments_delta("call_123", 'ry": "py')
    acc.add_arguments_delta("call_123", 'thon"}')

    tc = acc.complete_call("call_123")
    assert tc is not None
    assert tc.id == "call_123"
    assert tc.name == "search"
    assert tc.arguments == {"query": "python"}

    # Finalize all remaining
    acc.start_call("call_456", "get_time")
    acc.add_arguments_delta("call_456", "{}")
    remaining = acc.finalize_all()
    assert len(remaining) == 2
    assert remaining[1].id == "call_456"
    assert remaining[1].name == "get_time"


def test_sync_stream_response_lifecycle() -> None:
    chunks = [
        StreamChunk(text="First "),
        StreamChunk(text="second "),
        StreamChunk(text="third.", is_finished=True),
    ]

    on_first_token_called = []
    on_complete_called = []

    stream = StreamResponse(
        iter(chunks),
        on_first_token=lambda ttft: on_first_token_called.append(ttft),
        on_complete=lambda resp, ttft: on_complete_called.append(resp),
    )

    collected = []
    for c in stream:
        collected.append(str(c))

    assert "".join(collected) == "First second third."
    assert len(on_first_token_called) == 1
    assert len(on_complete_called) == 1

    final_resp = stream.response
    assert final_resp is not None
    assert final_resp.text == "First second third."

    # get_final_response returns same response
    assert stream.get_final_response() == final_resp


@pytest.mark.asyncio
async def test_async_stream_response_lifecycle() -> None:
    async def _chunk_gen():
        yield StreamChunk(text="Async ")
        yield StreamChunk(text="streaming ")
        yield StreamChunk(text="works!", is_finished=True)

    stream = AsyncStreamResponse(_chunk_gen())
    collected = []
    async for c in stream:
        collected.append(c.text)

    assert "".join(collected) == "Async streaming works!"
    final = await stream.get_final_response()
    assert final.text == "Async streaming works!"


# ==============================================================================
# OpenAI Provider Streaming Tests
# ==============================================================================


def test_openai_provider_generate_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()

    # Stream events: text delta, reasoning delta, tool call, completed
    event1 = MagicMock(type="response.output_text.delta", delta="Streaming ")
    event2 = MagicMock(type="response.output_text.delta", delta="text")
    event3 = MagicMock(
        type="response.completed",
        response=MagicMock(
            id="resp_123",
            output_text="Streaming text",
            output=[],
            usage=MagicMock(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    )

    mock_client.responses.create.return_value = [event1, event2, event3]
    mock_openai.OpenAI = MagicMock(return_value=mock_client)
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", api_key="sk-test")
    request = GenerationRequest(prompt="Hello")

    stream = provider.generate(request, stream=True)
    assert isinstance(stream, StreamResponse)

    parts = [chunk.text for chunk in stream]
    assert "".join(parts) == "Streaming text"

    resp = stream.response
    assert resp is not None
    assert resp.text == "Streaming text"
    assert resp.id == "resp_123"
    mock_client.responses.create.assert_called_once_with(
        model="gpt-6-astra",
        input=[{"role": "user", "content": "Hello"}],
        stream=True,
    )


@pytest.mark.asyncio
async def test_openai_provider_generate_async_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()

    class MockAsyncStream:
        def __init__(self, events):
            self.events = events

        def __aiter__(self):
            self._iter = iter(self.events)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    event1 = MagicMock(type="response.output_text.delta", delta="Async ")
    event2 = MagicMock(type="response.output_text.delta", delta="OpenAI")
    mock_client.responses.create = AsyncMock(
        return_value=MockAsyncStream([event1, event2])
    )
    mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
    mock_openai.OpenAI = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", api_key="sk-test")
    request = GenerationRequest(prompt="Hello async")

    stream = await provider.generate_async(request, stream=True)
    assert isinstance(stream, AsyncStreamResponse)

    parts = []
    async for chunk in stream:
        parts.append(chunk.text)

    assert "".join(parts) == "Async OpenAI"
    final = await stream.get_final_response()
    assert final.text == "Async OpenAI"


# ==============================================================================
# Anthropic Provider Streaming Tests
# ==============================================================================


def test_anthropic_provider_generate_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()

    # Content block deltas
    e1 = MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="Claude "))
    e2 = MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="stream"))
    e3 = MagicMock(type="message_stop")

    mock_client.messages.create.return_value = [e1, e2, e3]
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-7-sonnet", api_key="ant-test")
    request = GenerationRequest(prompt="Hello Claude")

    stream = provider.generate(request, stream=True)
    assert isinstance(stream, StreamResponse)

    parts = [str(c) for c in stream]
    assert "".join(parts) == "Claude stream"
    final = stream.get_final_response()
    assert final.text == "Claude stream"


# ==============================================================================
# Gemini Provider Streaming Tests
# ==============================================================================


def test_gemini_provider_generate_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")
    mock_types.Schema = MagicMock()

    mock_client = MagicMock()
    # Step delta text event
    text_delta = MagicMock(type="text", text="Gemini ")
    step_delta_event = MagicMock(event_type="step.delta", delta=text_delta)

    # Completed event
    mock_interaction = MagicMock(id="int_123", output_text="Gemini stream", steps=[], usage=None)
    completed_event = MagicMock(event_type="interaction.completed", interaction=mock_interaction)

    mock_client.interactions.create.return_value = [step_delta_event, completed_event]
    mock_genai.Client = MagicMock(return_value=mock_client)
    mock_genai.types = mock_types
    mock_google.genai = mock_genai

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="gem-test")
    request = GenerationRequest(prompt="Hello Gemini")

    stream = provider.generate(request, stream=True)
    assert isinstance(stream, StreamResponse)

    parts = [chunk.text for chunk in stream]
    assert "Gemini " in parts
    final = stream.get_final_response()
    assert final.text == "Gemini stream"
    assert final.id == "int_123"


def test_gemini_streaming_thought_summary_and_thinking_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()

    # Events sequence:
    # 1. interaction.created
    int_created_event = {
        "event_type": "interaction.created",
        "interaction": {"id": "int_thought_stream", "model": "gemini-3.8-flash"},
    }
    # 2. step.start thought
    thought_start_event = {
        "event_type": "step.start",
        "index": 0,
        "step": {"type": "thought"},
    }
    # 3. step.delta thought_summary (nested content.text)
    thought_delta_event = {
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "thought_summary",
            "content": {"type": "text", "text": "Analyzing problem logically..."},
        },
    }
    # 4. step.delta thought_signature
    sig_delta_event = {
        "event_type": "step.delta",
        "index": 0,
        "delta": {"type": "thought_signature", "signature": "sig_thought_123"},
    }
    # 5. step.stop thought
    thought_stop_event = {"event_type": "step.stop", "index": 0}
    # 6. step.start model_output
    output_start_event = {
        "event_type": "step.start",
        "index": 1,
        "step": {"type": "model_output"},
    }
    # 7. step.delta text
    output_delta_event = {
        "event_type": "step.delta",
        "index": 1,
        "delta": {"type": "text", "text": "42"},
    }
    # 8. step.stop model_output
    output_stop_event = {"event_type": "step.stop", "index": 1}
    # 9. interaction.completed (SSE payload omits steps)
    mock_interaction = MagicMock(id="int_thought_stream", output_text="", steps=[], usage=None)
    completed_event = MagicMock(event_type="interaction.completed", interaction=mock_interaction)

    mock_client.interactions.create.return_value = [
        int_created_event,
        thought_start_event,
        thought_delta_event,
        sig_delta_event,
        thought_stop_event,
        output_start_event,
        output_delta_event,
        output_stop_event,
        completed_event,
    ]
    mock_genai.Client = MagicMock(return_value=mock_client)
    mock_genai.types = mock_types
    mock_google.genai = mock_genai

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="gem-test")
    stream = provider.generate(GenerationRequest(prompt="What is the answer?"), stream=True)

    chunks = list(stream)
    thinking_chunks = [c.thinking for c in chunks if c.thinking]
    assert thinking_chunks == ["Analyzing problem logically..."]

    text_chunks = [c.text for c in chunks if c.text]
    assert text_chunks == ["42"]

    final = stream.get_final_response()
    assert final.id == "int_thought_stream"
    assert final.text == "42"
    assert final.thinking == "Analyzing problem logically..."
    assert len(final.thought_blocks) >= 1
    assert any(tb.get("thought_signature") == "sig_thought_123" for tb in final.thought_blocks)


def test_gemini_streaming_tool_call_with_event_id_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()

    # Step.start with real function call id
    step_start = {
        "event_type": "step.start",
        "index": 0,
        "step": {"type": "function_call", "id": "call_weather_99", "name": "get_weather"},
    }
    # Step.delta has event_id token (e.g. evt_resumption_token) which should NOT hijack call_id
    step_delta = {
        "event_type": "step.delta",
        "index": 0,
        "event_id": "evt_resumption_token_xyz",
        "delta": {"type": "arguments_delta", "arguments": '{"city": "Paris"}'},
    }
    # Step.stop also has event_id
    step_stop = {
        "event_type": "step.stop",
        "index": 0,
        "event_id": "evt_resumption_token_xyz",
    }
    completed = {
        "event_type": "interaction.completed",
        "interaction": {"id": "int_tool_chain", "output_text": "", "steps": []},
    }

    mock_client.interactions.create.return_value = [step_start, step_delta, step_stop, completed]
    mock_genai.Client = MagicMock(return_value=mock_client)
    mock_genai.types = mock_types
    mock_google.genai = mock_genai

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="gem-test")
    stream = provider.generate(GenerationRequest(prompt="Paris weather?"), stream=True)

    chunks = list(stream)
    tool_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_chunks) == 1
    tc = tool_chunks[0].tool_calls[0]
    assert tc.id == "call_weather_99"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}

    final = stream.get_final_response()
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].id == "call_weather_99"


def test_gemini_streaming_error_event_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()
    error_event = {
        "event_type": "error",
        "error": {"code": "resource_exhausted", "message": "Rate limit exceeded"},
    }
    mock_client.interactions.create.return_value = [error_event]
    mock_genai.Client = MagicMock(return_value=mock_client)
    mock_genai.types = mock_types
    mock_google.genai = mock_genai

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="gem-test")
    stream = provider.generate(GenerationRequest(prompt="Hello"), stream=True)

    with pytest.raises(RuntimeError, match="Gemini streaming error: Rate limit exceeded"):
        list(stream)


def test_gemini_streaming_finalize_in_flight_calls() -> None:
    from evidor.agent.providers.gemini.parser import GeminiResponseParser
    from evidor.agent.providers.gemini.stream import GeminiStreamHandler

    parser = GeminiResponseParser(model="gemini-3.8-flash")
    handler = GeminiStreamHandler(parser)

    # In-flight call started and partially accumulated, but stream ends before step.stop
    list(handler.process_event({
        "event_type": "step.start",
        "index": 0,
        "step": {"type": "function_call", "id": "call_in_flight", "name": "do_work"},
    }))
    list(handler.process_event({
        "event_type": "step.delta",
        "index": 0,
        "delta": {"type": "arguments_delta", "arguments": '{"status": "ok"}'},
    }))

    remaining = list(handler.finalize())
    assert len(remaining) == 1
    assert remaining[0].tool_calls[0].id == "call_in_flight"
    assert remaining[0].tool_calls[0].arguments == {"status": "ok"}



# ==============================================================================
# Agent Streaming & Telemetry Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_send_async_streaming_text_and_telemetry() -> None:
    # Setup mock provider yielding StreamChunks
    mock_provider = MagicMock()
    mock_provider.context_strategy = "stateless_replay"
    mock_provider.model = "test-model"

    async def _async_gen(req, stream=False):
        async def _chunks():
            yield StreamChunk(text="Live ")
            yield StreamChunk(text="streaming ")
            yield StreamChunk(text="response.", is_finished=True)

        return AsyncStreamResponse(_chunks())

    mock_provider.generate_async = _async_gen

    sink = InMemorySink()
    runtime = TelemetryRuntime([sink])
    agent = Agent(provider=mock_provider, telemetry=runtime)

    stream = await agent.send_async("Tell me a story", stream=True)
    assert isinstance(stream, AsyncStreamResponse)

    collected = []
    async for chunk in stream:
        collected.append(str(chunk))

    assert "".join(collected) == "Live streaming response."

    # History should be updated with user and assistant messages
    assert len(agent.messages) == 2
    assert agent.messages[0].role == "user"
    assert agent.messages[0].content == "Tell me a story"
    assert agent.messages[1].role == "assistant"
    assert agent.messages[1].content == "Live streaming response."

    # Close agent to flush telemetry runtime
    agent.close()

    # Verify telemetry events
    start_events = [e for e in sink.events if isinstance(e, LLMCallStartEvent)]
    end_events = [e for e in sink.events if isinstance(e, LLMCallEndEvent)]

    assert len(start_events) == 1
    assert start_events[0].is_streaming is True
    assert len(end_events) == 1
    assert end_events[0].is_streaming is True
    assert end_events[0].output_text == "Live streaming response."
    assert end_events[0].time_to_first_token_ms is not None
    assert end_events[0].time_to_first_token_ms >= 0.0


def test_agent_sync_send_streaming() -> None:
    mock_provider = MagicMock()
    mock_provider.context_strategy = "stateless_replay"
    mock_provider.model = "test-model"

    async def _async_gen(req, stream=False):
        async def _chunks():
            yield StreamChunk(text="Sync ")
            yield StreamChunk(text="stream ")
            yield StreamChunk(text="worked!", is_finished=True)

        return AsyncStreamResponse(_chunks())

    mock_provider.generate_async = _async_gen

    agent = Agent(provider=mock_provider)
    stream = agent.send("Hello sync", stream=True)
    assert isinstance(stream, StreamResponse)

    output = "".join(str(chunk) for chunk in stream)
    assert output == "Sync stream worked!"

    assert len(agent.messages) == 2
    assert agent.messages[1].content == "Sync stream worked!"
    agent.close()


@pytest.mark.asyncio
async def test_agent_streaming_with_tool_execution() -> None:
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather in {location}: Sunny 72F"

    mock_provider = MagicMock()
    mock_provider.context_strategy = "stateless_replay"
    mock_provider.model = "test-model"

    turn_count = 0

    async def _mock_generate_async(req, stream=False):
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            # First turn: returns tool call
            async def _t1():
                tc = ToolCall(id="c1", name="get_weather", arguments={"location": "Tokyo"})
                yield StreamChunk(thinking="Checking weather...", tool_calls=(tc,), is_finished=True)

            return AsyncStreamResponse(_t1())
        else:
            # Second turn: returns final text
            async def _t2():
                yield StreamChunk(text="It is ")
                yield StreamChunk(text="Sunny 72F in Tokyo.", is_finished=True)

            return AsyncStreamResponse(_t2())

    mock_provider.generate_async = _mock_generate_async

    agent = Agent(provider=mock_provider, tools=[get_weather])

    stream = await agent.send_async("What is the weather in Tokyo?", stream=True)
    collected = []
    async for chunk in stream:
        if chunk.text:
            collected.append(chunk.text)

    assert "".join(collected) == "It is Sunny 72F in Tokyo."

    # Check conversation history contains user -> assistant (tool call) -> tool result -> assistant (answer)
    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert agent.messages[2].content == "Weather in Tokyo: Sunny 72F"
    assert agent.messages[3].content == "It is Sunny 72F in Tokyo."


# ==============================================================================
# OpenAI Streaming Handler Tests (3.x fixes)
# ==============================================================================


def test_openai_streaming_tool_call_item_id_mapping() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser
    from evidor.agent.providers.openai.stream import OpenAIStreamHandler

    parser = OpenAIResponseParser(model="gpt-6-astra")
    handler = OpenAIStreamHandler(parser)

    # 1. Output item added with item.id ("fc_001") and item.call_id ("call_abc")
    item_added = {
        "type": "response.output_item.added",
        "item": {
            "id": "fc_001",
            "call_id": "call_abc",
            "type": "function_call",
            "name": "lookup_weather",
        },
    }
    chunks1 = list(handler.process_event(item_added))
    assert chunks1 == []
    assert handler.item_id_map.get("fc_001") == "call_abc"

    # 2. Argument deltas carry item_id ("fc_001") WITHOUT call_id
    delta1 = {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_001",
        "delta": '{"location":',
    }
    delta2 = {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_001",
        "delta": ' "Tokyo"}',
    }
    chunks2 = list(handler.process_event(delta1))
    chunks3 = list(handler.process_event(delta2))
    assert chunks2 == []
    assert chunks3 == []

    # 3. Arguments done event carries item_id ("fc_001")
    args_done = {
        "type": "response.function_call_arguments.done",
        "item_id": "fc_001",
    }
    chunks4 = list(handler.process_event(args_done))
    assert len(chunks4) == 1
    assert len(chunks4[0].tool_calls) == 1
    tc = chunks4[0].tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "lookup_weather"
    assert tc.arguments == {"location": "Tokyo"}

    # 4. output_item.done should NOT re-emit the already completed call
    item_done = {
        "type": "response.output_item.done",
        "item": {
            "id": "fc_001",
            "call_id": "call_abc",
            "type": "function_call",
            "name": "lookup_weather",
        },
    }
    chunks5 = list(handler.process_event(item_done))
    assert chunks5 == []

    # 5. finalize() must NOT double-yield already completed calls
    final_chunks = list(handler.finalize())
    assert final_chunks == []


def test_openai_streaming_finalize_in_flight_calls() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser
    from evidor.agent.providers.openai.stream import OpenAIStreamHandler

    parser = OpenAIResponseParser(model="gpt-6-astra")
    handler = OpenAIStreamHandler(parser)

    # Start a call that does not receive a completion event before stream ends
    item_added = {
        "type": "response.output_item.added",
        "item": {
            "id": "fc_in_flight",
            "call_id": "call_in_flight",
            "type": "function_call",
            "name": "search_db",
        },
    }
    list(handler.process_event(item_added))

    delta = {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_in_flight",
        "delta": '{"query": "antigravity"}',
    }
    list(handler.process_event(delta))

    # finalize completes in-flight calls
    final_chunks = list(handler.finalize())
    assert len(final_chunks) == 1
    assert len(final_chunks[0].tool_calls) == 1
    tc = final_chunks[0].tool_calls[0]
    assert tc.id == "call_in_flight"
    assert tc.name == "search_db"
    assert tc.arguments == {"query": "antigravity"}

    # Subsequent finalize yields nothing
    assert list(handler.finalize()) == []


def test_openai_streaming_refusal_delta() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser
    from evidor.agent.providers.openai.stream import OpenAIStreamHandler

    parser = OpenAIResponseParser(model="gpt-6-astra")
    handler = OpenAIStreamHandler(parser)

    refusal_event = {
        "type": "response.refusal.delta",
        "delta": "I cannot perform that request.",
    }
    chunks = list(handler.process_event(refusal_event))
    assert len(chunks) == 1
    assert chunks[0].text == "I cannot perform that request."


def test_openai_streaming_error_events_raise() -> None:
    import pytest
    from evidor.agent.providers.openai.parser import OpenAIResponseParser
    from evidor.agent.providers.openai.stream import OpenAIStreamHandler

    parser = OpenAIResponseParser(model="gpt-6-astra")
    handler = OpenAIStreamHandler(parser)

    err_event = {
        "type": "response.error",
        "code": "rate_limit_exceeded",
        "message": "Quota exceeded",
    }
    with pytest.raises(RuntimeError, match="OpenAI stream error \\(rate_limit_exceeded\\): Quota exceeded"):
        list(handler.process_event(err_event))

    fail_event = {
        "type": "response.failed",
        "response": {
            "error": {
                "code": "server_error",
                "message": "Internal error occurred",
            }
        },
    }
    with pytest.raises(RuntimeError, match="OpenAI response failed \\(server_error\\): Internal error occurred"):
        list(handler.process_event(fail_event))

