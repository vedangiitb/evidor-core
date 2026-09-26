import pytest

from evidor import Agent, GenerationRequest, GenerationResponse, Message, ModelProvider


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
