import asyncio
import pytest

from evidor import (
    Agent,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelProvider,
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


def test_default_summarization_uses_main_provider() -> None:
    provider = FakeProvider(model="primary-model")
    agent = Agent(provider, max_messages=3)

    agent.send("turn 1")
    agent.send("turn 2")
    agent.send("turn 3")

    # Compaction occurred: agent messages include a summary message
    summary_messages = [m for m in agent.messages if m.is_summary]
    assert len(summary_messages) == 1
    assert "primary-model" in summary_messages[0].content


def test_custom_summarization_model_as_string() -> None:
    provider = FakeProvider(model="primary-model")
    agent = Agent(provider, max_messages=3, summarization_model="cheap-summarizer")

    assert agent.summary_provider.model == "cheap-summarizer"

    agent.send("turn 1")
    agent.send("turn 2")
    agent.send("turn 3")

    summary_messages = [m for m in agent.messages if m.is_summary]
    assert len(summary_messages) == 1
    assert "cheap-summarizer" in summary_messages[0].content


def test_custom_summarization_model_as_provider_instance() -> None:
    main_provider = FakeProvider(model="heavy-model")
    summary_provider = FakeProvider(model="light-model")

    agent = Agent(main_provider, max_messages=3, summarization_model=summary_provider)

    assert agent.summary_provider is summary_provider

    agent.send("turn 1")
    agent.send("turn 2")
    agent.send("turn 3")

    # summary_provider received the summary request
    assert len(summary_provider.requests) == 1
    assert summary_provider.requests[0].messages[0].role == "system"
    assert "Summarize" in summary_provider.requests[0].messages[0].content

    # main_provider handled conversation turns
    summary_messages = [m for m in agent.messages if m.is_summary]
    assert len(summary_messages) == 1
    assert "light-model" in summary_messages[0].content


def test_summarization_model_fallback_with_api_key() -> None:
    class ProviderWithKeyNoWithModel:
        def __init__(self, model: str, api_key: str | None = None) -> None:
            self.model = model
            self._api_key = api_key

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(text="ok", model=self.model)

    provider = ProviderWithKeyNoWithModel(model="main", api_key="secret-key")
    agent = Agent(provider, summarization_model="summary-model")
    assert agent.summary_provider.model == "summary-model"
    assert agent.summary_provider._api_key == "secret-key"


def test_summarization_model_fallback_without_api_key() -> None:
    class ProviderWithoutKeyNoWithModel:
        def __init__(self, model: str) -> None:
            self.model = model

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(text="ok", model=self.model)

    provider = ProviderWithoutKeyNoWithModel(model="main")
    agent = Agent(provider, summarization_model="summary-model")
    assert agent.summary_provider.model == "summary-model"


def test_summarization_model_fallback_incompatible_raises() -> None:
    class IncompatibleProvider:
        def __init__(self, fixed_arg: int) -> None:
            self.fixed = fixed_arg

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(text="ok", model="fixed")

    provider = IncompatibleProvider(fixed_arg=42)
    with pytest.raises(ValueError, match="Cannot configure summarization_model string 'summary-model'"):
        Agent(provider, summarization_model="summary-model")


def test_summarization_model_validation() -> None:
    provider = FakeProvider()

    with pytest.raises(ValueError, match="summarization_model must not be empty"):
        Agent(provider, summarization_model="   ")

    with pytest.raises(TypeError, match="summarization_model must be a model name string or ModelProvider"):
        Agent(provider, summarization_model=123)  # type: ignore[arg-type]


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


def test_agent_summarization_with_tools() -> None:
    @tool
    def ping() -> str:
        return "pong"

    summaries = []

    class SummarizeToolProvider:
        def with_model(self, model: str) -> "SummarizeToolProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            # Check if this is a summarization request
            if any(m.role == "system" and "Summarize the conversation" in m.content for m in request.messages):
                summaries.append(request.messages[-1].content)
                return GenerationResponse(text="Summary containing tool results", model="test")

            # Normal turn: return text without tool call
            return GenerationResponse(text=f"Reply to: {request.prompt}", model="test")

    provider = SummarizeToolProvider()
    agent = Agent(provider, tools=[ping], max_messages=3)

    # Manually inject tool message into history to test transcript formatting in _summarize
    agent._messages.append(Message(role="assistant", content="calling ping", tool_calls=(ToolCall(id="c", name="ping", arguments={}),)))
    agent._messages.append(Message(role="tool", content="pong", tool_call_id="c", name="ping"))

    agent.send("next message")
    agent.send("another message")

    assert len(summaries) >= 1
    summary_input = summaries[0]
    assert "TOOL (ping): pong" in summary_input
    assert "ASSISTANT (called tools: ping): calling ping" in summary_input


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


def test_summarize_with_empty_content_tool_call() -> None:
    tool = Tool(
        name="echo",
        description="Echo",
        parameters={"type": "object", "properties": {}},
        func=lambda: "done",
    )
    summary_provider = SequenceProvider([GenerationResponse(text="summary text", model="test")])
    agent = Agent(
        provider=FakeProvider(),
        summarization_model=summary_provider,
        tools=[tool],
    )
    messages = [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="c1", name="echo", arguments={}),),
        ),
        Message(role="tool", content="done", name="echo"),
    ]
    summary = agent._summarize(messages)
    assert summary == "summary text"
    assert "ASSISTANT (called tools: echo):" in summary_provider.requests[0].messages[1].content


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
