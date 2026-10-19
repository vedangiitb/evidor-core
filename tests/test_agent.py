import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock
import pytest

from evidor import (
    Agent,
    AnthropicProvider,
    GeminiProvider,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelProvider,
    OpenAIProvider,
    Tool,
    ToolCall,
    tool,
)


class FakeProvider:
    def __init__(self, model: str = "fake") -> None:
        self.model = model
        self.requests: list[GenerationRequest] = []

    def with_model(self, model: str) -> "FakeProvider":
        return FakeProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        return GenerationResponse(text=f"[{self.model}] {request.prompt}", model=self.model)

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate(request)


def test_agent_delegates_to_its_provider() -> None:
    agent = Agent(FakeProvider(model="default"))

    response = agent.send("hello")

    assert response == GenerationResponse(text="[default] hello", model="default")
    assert isinstance(FakeProvider(), ModelProvider)


def test_agent_with_system_prompt_and_clear_history() -> None:
    agent = Agent(FakeProvider(), system_prompt="You are a helper")

    # Initial message contains system prompt
    assert len(agent.messages) == 1
    assert agent.messages[0] == Message(role="system", content="You are a helper")

    agent.send("hello")
    assert len(agent.messages) == 3

    # Clear history preserves system prompt
    agent.clear_history()
    assert len(agent.messages) == 1
    assert agent.messages[0] == Message(role="system", content="You are a helper")


# --- Tool Calling Tests ---


def test_agent_automatic_tool_calling_loop() -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather."""
        return f"{location} is 25C and sunny"

    class ToolFakeProvider:
        def __init__(self) -> None:
            self.turn = 0

        def with_model(self, model: str) -> "ToolFakeProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.turn += 1
            if self.turn == 1:
                # First response calls the tool
                return GenerationResponse(
                    text="",
                    model="test",
                    tool_calls=(
                        ToolCall(id="call_1", name="get_weather", arguments={"location": "Paris"}),
                    ),
                )
            # Second response receives tool output and gives answer
            tool_msg = [m for m in request.messages if m.role == "tool"][0]
            return GenerationResponse(
                text=f"The weather is: {tool_msg.content}",
                model="test",
            )

    provider = ToolFakeProvider()
    agent = Agent(provider, tools=[get_weather])

    assert len(agent.tools) == 1
    assert agent.tools[0].name == "get_weather"

    response = agent.send("What's the weather in Paris?")

    assert response.text == "The weather is: Paris is 25C and sunny"
    assert len(agent.messages) == 4
    assert agent.messages[0].role == "user"
    assert agent.messages[1].role == "assistant"
    assert len(agent.messages[1].tool_calls) == 1
    assert agent.messages[2].role == "tool"
    assert agent.messages[2].content == "Paris is 25C and sunny"
    assert agent.messages[2].tool_call_id == "call_1"
    assert agent.messages[3].role == "assistant"


def test_agent_multiple_tool_calls_in_single_turn() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        return a * b

    class MultiToolProvider:
        def __init__(self) -> None:
            self.called = False

        def with_model(self, model: str) -> "MultiToolProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            if not self.called:
                self.called = True
                return GenerationResponse(
                    text="",
                    model="test",
                    tool_calls=(
                        ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
                        ToolCall(id="c2", name="multiply", arguments={"a": 4, "b": 5}),
                    ),
                )
            return GenerationResponse(text="Both tools completed", model="test")

    agent = Agent(MultiToolProvider(), tools=[add, multiply])
    response = agent.send("Compute both")

    assert response.text == "Both tools completed"
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].content == "5"
    assert tool_messages[1].content == "20"


def test_agent_handles_unknown_tool_and_exception() -> None:
    @tool
    def faulty_tool(x: int) -> int:
        raise ValueError("Something broke")

    class FaultyProvider:
        def __init__(self) -> None:
            self.turn = 0

        def with_model(self, model: str) -> "FaultyProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.turn += 1
            if self.turn == 1:
                return GenerationResponse(
                    text="",
                    model="test",
                    tool_calls=(
                        ToolCall(id="c1", name="nonexistent", arguments={}),
                        ToolCall(id="c2", name="faulty_tool", arguments={"x": 1}),
                    ),
                )
            return GenerationResponse(text="Handled errors", model="test")

    agent = Agent(FaultyProvider(), tools=[faulty_tool])
    response = agent.send("test")

    assert response.text == "Handled errors"
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert "Error: Tool 'nonexistent' not found." in tool_messages[0].content
    assert "Error executing tool 'faulty_tool': Something broke" in tool_messages[1].content


def test_agent_max_tool_iterations() -> None:
    @tool
    def ping() -> str:
        return "pong"

    class InfiniteToolProvider:
        def with_model(self, model: str) -> "InfiniteToolProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                text="",
                model="test",
                tool_calls=(ToolCall(id="c1", name="ping", arguments={}),),
            )

    agent = Agent(InfiniteToolProvider(), tools=[ping], max_tool_iterations=3)
    response = agent.send("loop")

    # Stopped after 3 iterations
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_messages) == 3


class SequenceProvider:
    def __init__(self, responses: list[GenerationResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[GenerationRequest] = []

    def with_model(self, model: str) -> "SequenceProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def test_tool_returning_custom_object() -> None:
    class CustomObj:
        def __str__(self) -> str:
            return "CustomResult"

    tool = Tool(
        name="custom_tool",
        description="Returns custom object",
        parameters={"type": "object", "properties": {}},
        func=lambda: CustomObj(),
    )
    provider = SequenceProvider(
        [
            GenerationResponse(
                text="",
                model="test",
                tool_calls=(ToolCall(id="c1", name="custom_tool", arguments={}),),
            ),
            GenerationResponse(text="final reply", model="test"),
        ]
    )
    agent = Agent(provider=provider, tools=[tool])
    agent.send("run")
    assert agent.messages[-2].role == "tool"
    assert agent.messages[-2].content == "CustomResult"


def test_agent_handles_malformed_json_arguments() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    provider = SequenceProvider(
        [
            GenerationResponse(
                text="",
                model="test",
                tool_calls=(
                    ToolCall(
                        id="call_err",
                        name="add",
                        arguments={"__decode_error__": "Invalid JSON syntax", "__raw_args__": "{a: 1,"},
                    ),
                ),
            ),
            GenerationResponse(text="Recovered from malformed JSON", model="test"),
        ]
    )

    agent = Agent(provider=provider, tools=[add])
    resp = agent.send("add numbers")

    assert resp.text == "Recovered from malformed JSON"
    tool_msg = [m for m in agent.messages if m.role == "tool"][0]
    assert "Error: Malformed JSON arguments for tool 'add': Invalid JSON syntax." in tool_msg.content
    assert "Received input: {a: 1," in tool_msg.content


def test_agent_max_iterations_final_synthesis() -> None:
    @tool
    def search(q: str) -> str:
        return f"result for {q}"

    class SynthesisProvider:
        def __init__(self) -> None:
            self.calls = 0

        def with_model(self, model: str) -> "SynthesisProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.calls += 1
            if request.tools:
                # Still in tool loop
                return GenerationResponse(
                    text="",
                    model="test",
                    tool_calls=(ToolCall(id=f"c{self.calls}", name="search", arguments={"q": f"step {self.calls}"}),),
                )
            # Final synthesis called with tools=()
            return GenerationResponse(text="Final synthesized answer from all tools", model="test")

    agent = Agent(SynthesisProvider(), tools=[search], max_tool_iterations=2)
    response = agent.send("find answer")

    assert response.text == "Final synthesized answer from all tools"
    assert agent.messages[-1].content == "Final synthesized answer from all tools"


def test_agent_tool_timeout() -> None:
    import time

    @tool
    def slow_tool() -> str:
        time.sleep(0.1)
        return "finished"

    provider = SequenceProvider(
        [
            GenerationResponse(
                text="",
                model="test",
                tool_calls=(ToolCall(id="c1", name="slow_tool", arguments={}),),
            ),
            GenerationResponse(text="Tool timed out", model="test"),
        ]
    )

    agent = Agent(provider, tools=[slow_tool], tool_timeout=0.03)
    assert agent.tool_timeout == 0.03
    agent.send("run slow")

    tool_msg = [m for m in agent.messages if m.role == "tool"][0]
    assert "Error executing tool 'slow_tool': Tool 'slow_tool' timed out after 0.03s" in tool_msg.content


def test_agent_send_async_basic() -> None:
    async def _run() -> None:
        provider = FakeProvider(model="test-model")
        agent = Agent(provider, system_prompt="Sys prompt")
        resp = await agent.send_async("hello async")
        assert resp.text == "[test-model] hello async"
        assert len(agent.messages) == 3

    asyncio.run(_run())


def test_agent_send_async_with_tools() -> None:
    @tool
    async def async_fetch(item: str) -> str:
        await asyncio.sleep(0.01)
        return f"Fetched {item}"

    provider = SequenceProvider(
        [
            GenerationResponse(
                text="Looking up...",
                model="test",
                tool_calls=(ToolCall(id="c1", name="async_fetch", arguments={"item": "data"}),),
            ),
            GenerationResponse(text="Here is your data: Fetched data", model="test"),
        ]
    )

    agent = Agent(provider, tools=[async_fetch])

    async def _run() -> None:
        resp = await agent.send_async("get data")
        assert resp.text == "Here is your data: Fetched data"
        assert len(agent.messages) == 4
        assert agent.messages[2].role == "tool"
        assert agent.messages[2].content == "Fetched data"

    asyncio.run(_run())


def test_agent_send_async_errors_and_expiry() -> None:
    @tool
    def faulty_func() -> str:
        raise ValueError("broken")

    class ExpiryProvider:
        def __init__(self) -> None:
            self.turn = 0

        def with_model(self, model: str) -> "ExpiryProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.turn += 1
            if request.tools:
                if self.turn == 1:
                    # Malformed JSON
                    return GenerationResponse(
                        text="",
                        model="test",
                        tool_calls=(
                            ToolCall(id="c1", name="faulty_func", arguments={"__decode_error__": "bad syntax"}),
                        ),
                    )
                elif self.turn == 2:
                    # Unknown tool
                    return GenerationResponse(
                        text="",
                        model="test",
                        tool_calls=(ToolCall(id="c2", name="missing_tool", arguments={}),),
                    )
                # Exception in tool
                return GenerationResponse(
                    text="",
                    model="test",
                    tool_calls=(ToolCall(id="c3", name="faulty_func", arguments={}),),
                )
            return GenerationResponse(text="", model="test")

    agent = Agent(ExpiryProvider(), tools=[faulty_func], max_tool_iterations=3)

    async def _run() -> None:
        resp = await agent.send_async("trigger errors")
        assert "Reached maximum tool iterations (3)" in resp.text
        tool_messages = [m for m in agent.messages if m.role == "tool"]
        assert len(tool_messages) == 3
        assert "Malformed JSON arguments" in tool_messages[0].content
        assert "Tool 'missing_tool' not found" in tool_messages[1].content
        assert "Error executing tool 'faulty_func': broken" in tool_messages[2].content

    asyncio.run(_run())


def test_agent_handles_empty_assistant_response_gracefully() -> None:
    class EmptyResponseProvider:
        def __init__(self) -> None:
            self.model = "empty-provider"

        def with_model(self, model: str) -> "EmptyResponseProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(text="", model=self.model)

    agent = Agent(EmptyResponseProvider())
    resp = agent.send("hello")
    assert resp.text == ""
    assert agent.messages[-1].content == "Done."


def test_agent_sync_methods_raise_in_running_event_loop() -> None:
    agent = Agent(FakeProvider())

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="Agent.send\\(\\) cannot be called from a running event loop"):
            agent.send("test")

    asyncio.run(_run())


def test_agent_send_async_concurrent_tools() -> None:
    execution_order: list[str] = []

    @tool
    async def fast_tool() -> str:
        await asyncio.sleep(0.01)
        execution_order.append("fast")
        return "fast_done"

    @tool
    async def slow_tool() -> str:
        await asyncio.sleep(0.03)
        execution_order.append("slow")
        return "slow_done"

    class MultiToolProvider:
        def __init__(self) -> None:
            self.turn = 0
            self.model = "multi-tool"

        def with_model(self, model: str) -> "MultiToolProvider":
            return self

        async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
            self.turn += 1
            if self.turn == 1:
                return GenerationResponse(
                    text="",
                    model="multi-tool",
                    tool_calls=(
                        ToolCall(id="c1", name="slow_tool", arguments={}),
                        ToolCall(id="c2", name="fast_tool", arguments={}),
                    ),
                )
            return GenerationResponse(text="Both tools completed", model="multi-tool")

    agent = Agent(MultiToolProvider(), tools=[slow_tool, fast_tool])

    async def _run() -> None:
        resp = await agent.send_async("run both")
        assert resp.text == "Both tools completed"
        # Since slow_tool takes 30ms and fast_tool takes 10ms and they run concurrently,
        # fast_tool finishes first!
        assert execution_order == ["fast", "slow"]
        tool_msgs = [m for m in agent.messages if m.role == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == "slow_done"
        assert tool_msgs[1].content == "fast_done"

    asyncio.run(_run())


def test_providers_generate_raise_inside_running_event_loop() -> None:
    openai_p = OpenAIProvider(model="gpt-4.1-mini")
    anthropic_p = AnthropicProvider(model="claude-3-5")
    gemini_p = GeminiProvider(model="gemini-3.8-flash")
    req = GenerationRequest(prompt="hi")

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="OpenAIProvider.generate\\(\\) cannot be called from a running event loop"):
            openai_p.generate(req)

        with pytest.raises(RuntimeError, match="AnthropicProvider.generate\\(\\) cannot be called from a running event loop"):
            anthropic_p.generate(req)

        with pytest.raises(RuntimeError, match="GeminiProvider.generate\\(\\) cannot be called from a running event loop"):
            gemini_p.generate(req)

    asyncio.run(_run())


def test_agent_sync_send_reuses_background_loop() -> None:
    captured_loops: list[asyncio.AbstractEventLoop] = []

    class LoopTrackingProvider:
        def __init__(self) -> None:
            self.model = "tracker"

        def with_model(self, model: str) -> "LoopTrackingProvider":
            return self

        async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
            captured_loops.append(asyncio.get_running_loop())
            return GenerationResponse(text="tracked", model="tracker")

    agent = Agent(LoopTrackingProvider())

    assert agent._background_loop is None

    agent.send("turn 1")
    agent.send("turn 2")

    # Both turns ran on the exact same persistent background event loop
    assert len(captured_loops) == 2
    assert captured_loops[0] is captured_loops[1]
    assert agent._background_loop is not None
    assert agent._background_loop.is_running

    agent.close()
    assert agent._background_loop is None


def test_background_event_loop_lifecycle() -> None:
    from evidor.utils import BackgroundEventLoop

    bg = BackgroundEventLoop(name="TestLoop")
    assert not bg.is_running

    async def add(a: int, b: int) -> int:
        return a + b

    res1 = bg.run(lambda: add(1, 2))
    assert res1 == 3
    assert bg.is_running

    res2 = bg.run(lambda: add(3, 4))
    assert res2 == 7

    bg.close()
    assert not bg.is_running

    with pytest.raises(RuntimeError, match="BackgroundEventLoop has been closed"):
        bg.run(lambda: add(5, 6))


def test_agent_send_then_close_async() -> None:
    provider = FakeProvider(model="test")
    provider.close_async = AsyncMock()
    provider.close = MagicMock()

    agent = Agent(provider)
    res = agent.send("hello")
    assert res.text == "[test] hello"
    assert agent._background_loop is not None
    assert agent._background_loop.is_running

    async def _close():
        await agent.close_async()

    asyncio.run(_close())

    assert agent._background_loop is None
    provider.close_async.assert_awaited_once()
    provider.close.assert_called_once()


def test_agent_close_without_background_loop_closes_async_resources() -> None:
    provider = FakeProvider(model="test")
    provider.close_async = AsyncMock()
    provider.close = MagicMock()

    agent = Agent(provider)
    assert agent._background_loop is None

    agent.close()

    provider.close_async.assert_awaited_once()
    provider.close.assert_called_once()


def test_agent_concurrent_sync_send() -> None:
    provider = FakeProvider(model="concurrent")
    agent = Agent(provider)

    errors = []

    def _worker(msg: str):
        try:
            agent.send(msg)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(f"msg_{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # 10 user messages and 10 assistant messages
    assert len(agent.messages) == 20
    agent.close()


