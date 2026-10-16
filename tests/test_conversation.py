import asyncio
import pytest

from evidor import (
    Agent,
    AnthropicProvider,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelProvider,
    ToolCall,
    tool,
)


class StatefulChainProvider:
    """Mock provider with 'response_chain' context strategy (like OpenAI Responses API)."""

    def __init__(self, model: str = "gpt-6-astra") -> None:
        self.model = model
        self.context_strategy = "response_chain"
        self.requests: list[GenerationRequest] = []
        self._turn = 0

    def with_model(self, model: str) -> "StatefulChainProvider":
        return StatefulChainProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        self._turn += 1
        resp_id = f"resp_{self._turn}"
        return GenerationResponse(
            id=resp_id,
            text=f"Turn {self._turn} response for: {request.prompt}",
            model=self.model,
        )


class InteractionChainProvider:
    """Mock provider with 'interaction_chain' context strategy (like Gemini Interactions API)."""

    def __init__(self, model: str = "gemini-3.8-flash") -> None:
        self.model = model
        self.context_strategy = "interaction_chain"
        self.requests: list[GenerationRequest] = []
        self._turn = 0

    def with_model(self, model: str) -> "InteractionChainProvider":
        return InteractionChainProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        self._turn += 1
        resp_id = f"int_{self._turn}"
        return GenerationResponse(
            id=resp_id,
            text=f"Gemini turn {self._turn}: {request.prompt}",
            model=self.model,
        )


class StatelessReplayProvider:
    """Mock provider with 'stateless_replay' context strategy (like Anthropic)."""

    def __init__(self, model: str = "claude-3-7-sonnet-20250219") -> None:
        self.model = model
        self.context_strategy = "stateless_replay"
        self.requests: list[GenerationRequest] = []
        self._turn = 0

    def with_model(self, model: str) -> "StatelessReplayProvider":
        return StatelessReplayProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        self._turn += 1
        resp_id = f"msg_{self._turn}"
        return GenerationResponse(
            id=resp_id,
            text=f"Claude turn {self._turn}: {request.prompt}",
            model=self.model,
        )


def test_provider_managed_response_chain_multi_turn() -> None:
    provider = StatefulChainProvider()
    agent = Agent(provider, system_prompt="You are a helpful assistant.")

    assert agent.strategy == "response_chain"
    assert agent.last_response_id is None

    # First turn
    resp1 = agent.send("First question")
    assert resp1.id == "resp_1"
    assert agent.last_response_id == "resp_1"

    req1 = provider.requests[0]
    assert req1.previous_response_id is None
    assert req1.system == "You are a helpful assistant."
    assert len(req1.messages) == 1
    assert req1.messages[0].content == "First question"

    # Second turn: chains directly using previous_response_id
    resp2 = agent.send("Second question")
    assert resp2.id == "resp_2"
    assert agent.last_response_id == "resp_2"

    req2 = provider.requests[1]
    assert req2.previous_response_id == "resp_1"
    assert req2.system == "You are a helpful assistant."
    # In provider-managed chains, only the new turn's input is sent
    assert len(req2.messages) == 1
    assert req2.messages[0].content == "Second question"


def test_provider_managed_interaction_chain_multi_turn() -> None:
    provider = InteractionChainProvider()
    agent = Agent(provider, system_prompt="Gemini assistant")

    assert agent.strategy == "interaction_chain"
    assert agent.last_response_id is None

    # Turn 1
    resp1 = agent.send("Hello Gemini")
    assert resp1.id == "int_1"
    assert agent.last_response_id == "int_1"
    assert provider.requests[0].previous_response_id is None

    # Turn 2
    resp2 = agent.send("Tell me more")
    assert resp2.id == "int_2"
    assert agent.last_response_id == "int_2"
    assert provider.requests[1].previous_response_id == "int_1"


def test_stateless_replay_provider_multi_turn() -> None:
    provider = StatelessReplayProvider()
    agent = Agent(provider, system_prompt="Claude system")

    assert agent.strategy == "stateless_replay"

    resp1 = agent.send("Turn 1")
    assert resp1.id == "msg_1"
    assert len(provider.requests[0].messages) == 2  # system + user 1

    resp2 = agent.send("Turn 2")
    assert resp2.id == "msg_2"
    # Replays full accumulated history for stateless replay
    req2_msgs = provider.requests[1].messages
    assert len(req2_msgs) == 4  # system + user 1 + assistant 1 + user 2
    assert req2_msgs[-1].content == "Turn 2"


def test_anthropic_provider_strategy_and_properties() -> None:
    provider = AnthropicProvider(model="claude-3-7-sonnet-20250219")
    assert provider.context_strategy == "stateless_replay"
    assert provider.enable_prompt_caching is True

    agent = Agent(provider)
    assert agent.strategy == "stateless_replay"


def test_stateful_chain_tool_calling_loop() -> None:
    @tool
    def search_city(city: str) -> str:
        """Search for city info."""
        return f"{city}: population 2M"

    class ToolChainProvider:
        def __init__(self) -> None:
            self.model = "test-chain"
            self.context_strategy = "response_chain"
            self.requests: list[GenerationRequest] = []
            self.step = 0

        def with_model(self, model: str) -> "ToolChainProvider":
            return self

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            self.step += 1
            if self.step == 1:
                return GenerationResponse(
                    id="step_1",
                    text="",
                    model=self.model,
                    tool_calls=(ToolCall(id="call_c1", name="search_city", arguments={"city": "Vienna"}),),
                )
            return GenerationResponse(
                id="step_2",
                text=f"Result: {request.messages[0].content}",
                model=self.model,
            )

    provider = ToolChainProvider()
    agent = Agent(provider, tools=[search_city])

    response = agent.send("Tell me about Vienna")
    assert response.text == "Result: Vienna: population 2M"
    assert agent.last_response_id == "step_2"

    # Verify tool execution turn used previous_response_id from step_1
    req2 = provider.requests[1]
    assert req2.previous_response_id == "step_1"
    assert len(req2.messages) == 1
    assert req2.messages[0].role == "tool"
    assert req2.messages[0].content == "Vienna: population 2M"
    assert req2.messages[0].tool_call_id == "call_c1"


def test_clear_history_and_reset() -> None:
    provider = StatefulChainProvider()
    agent = Agent(provider, system_prompt="Persistent system")

    agent.send("Hello")
    assert agent.last_response_id == "resp_1"
    assert len(agent.messages) == 3  # system, user, assistant

    # clear_history resets provider state ID and cleans messages while preserving system prompt
    agent.clear_history()
    assert agent.last_response_id is None
    assert len(agent.messages) == 1
    assert agent.messages[0].role == "system"

    # Next send starts a fresh chain
    agent.send("New hello")
    assert provider.requests[1].previous_response_id is None
    assert agent.last_response_id == "resp_2"

    # reset() is alias for clear_history
    agent.reset()
    assert agent.last_response_id is None
    assert len(agent.messages) == 1


@pytest.mark.asyncio
async def test_async_stateful_chain() -> None:
    provider = StatefulChainProvider()
    agent = Agent(provider)

    resp1 = await agent.send_async("Async turn 1")
    assert resp1.id == "resp_1"
    assert agent.last_response_id == "resp_1"

    resp2 = await agent.send_async("Async turn 2")
    assert resp2.id == "resp_2"
    assert agent.last_response_id == "resp_2"
    assert provider.requests[1].previous_response_id == "resp_1"


class CompactingMockProvider:
    """Mock provider simulating native compaction emitting compaction summary."""

    def __init__(self, model: str = "claude-3-7-sonnet-20250219") -> None:
        self.model = model
        self.context_strategy = "stateless_replay"
        self._compact_threshold = None
        self.requests: list[GenerationRequest] = []
        self._turn = 0

    def with_model(self, model: str) -> "CompactingMockProvider":
        return CompactingMockProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        self._turn += 1
        compaction = "Compacted conversation state" if self._turn == 1 else None
        return GenerationResponse(
            id=f"msg_{self._turn}",
            text=f"Response {self._turn}",
            model=self.model,
            compaction=compaction,
        )


def test_agent_compaction_recording() -> None:
    provider = CompactingMockProvider()
    agent = Agent(provider, compact_threshold=150000)
    assert agent.compact_threshold == 150000
    assert provider._compact_threshold == 150000

    resp1 = agent.send("Turn 1 prompt")
    assert resp1.compaction == "Compacted conversation state"

    # Verify assistant message in agent.messages records the compaction
    assert len(agent.messages) == 2  # user, assistant
    asst_msg = agent.messages[1]
    assert asst_msg.role == "assistant"
    assert asst_msg.compaction == "Compacted conversation state"

    # On Turn 2, the replayed messages to the provider contain the compaction in the assistant message
    agent.send("Turn 2 prompt")
    assert len(provider.requests) == 2
    req2 = provider.requests[1]
    assert req2.messages[1].compaction == "Compacted conversation state"


