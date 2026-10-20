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
