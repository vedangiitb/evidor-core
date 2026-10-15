import sys
import types
from typing import Any
from unittest.mock import MagicMock
import pytest

from evidor import (
    Agent,
    AnthropicProvider,
    AnthropicThinkingConfig,
    ConsoleSink,
    GeminiProvider,
    GeminiThinkingConfig,
    GenerationRequest,
    GenerationResponse,
    InMemorySink,
    Message,
    OpenAIProvider,
    OpenAIThinkingConfig,
    ThinkingConfig,
    ToolCall,
)
from evidor.telemetry.events import (
    LLMCallEndEvent,
    TokenUsage,
    TraceContext,
    redact_event,
)
from evidor.telemetry import semconv


# ==============================================================================
# Configuration Classes Tests
# ==============================================================================


def test_thinking_config_base() -> None:
    cfg = ThinkingConfig(enabled=True)
    assert cfg.enabled is True
    cfg.validate()


def test_openai_thinking_config() -> None:
    cfg = OpenAIThinkingConfig(effort="high")
    assert cfg.effort == "high"
    assert cfg.enabled is True
    cfg.validate()

    # Coercion from shorthand string
    c1 = OpenAIThinkingConfig.from_value("low")
    assert c1.effort == "low"
    assert c1.enabled is True

    # Coercion from bool
    c2 = OpenAIThinkingConfig.from_value(False)
    assert c2.enabled is False

    # Identity
    c3 = OpenAIThinkingConfig.from_value(cfg)
    assert c3 is cfg

    # Invalid effort
    with pytest.raises(ValueError, match="Invalid OpenAI reasoning effort"):
        OpenAIThinkingConfig(effort="extreme").validate()  # type: ignore[arg-type]

    # Invalid type
    with pytest.raises(TypeError, match="Cannot coerce"):
        OpenAIThinkingConfig.from_value(1234)


def test_anthropic_thinking_config() -> None:
    cfg = AnthropicThinkingConfig(budget_tokens=4096)
    assert cfg.budget_tokens == 4096
    assert cfg.type == "enabled"
    cfg.validate()

    # Minimum budget constraint
    with pytest.raises(ValueError, match="Anthropic manual thinking budget_tokens must be >= 1024"):
        AnthropicThinkingConfig(budget_tokens=500).validate()

    # Adaptive effort
    adaptive = AnthropicThinkingConfig(type="adaptive", effort="high", budget_tokens=None)
    adaptive.validate()
    assert adaptive.type == "adaptive"
    assert adaptive.effort == "high"

    # Coercion from integer budget
    c_int = AnthropicThinkingConfig.from_value(2048)
    assert c_int.budget_tokens == 2048
    assert c_int.type == "enabled"

    # Coercion from effort string
    c_str = AnthropicThinkingConfig.from_value("high")
    assert c_str.type == "adaptive"
    assert c_str.effort == "high"

    # Coercion from bool
    c_bool = AnthropicThinkingConfig.from_value(False)
    assert c_bool.enabled is False

    # Invalid type
    with pytest.raises(TypeError, match="Cannot coerce"):
        AnthropicThinkingConfig.from_value({"dict": "invalid"})


def test_gemini_thinking_config() -> None:
    cfg = GeminiThinkingConfig(thinking_budget=2048, include_thoughts=True)
    assert cfg.thinking_budget == 2048
    assert cfg.include_thoughts is True
    cfg.validate()

    # Dynamic / None
    cfg_dyn = GeminiThinkingConfig(thinking_budget=None)
    cfg_dyn.validate()

    # Disabled
    cfg_off = GeminiThinkingConfig(thinking_budget=0)
    cfg_off.validate()

    # Invalid budget < -1
    with pytest.raises(ValueError, match="Gemini thinking_budget must be >= -1"):
        GeminiThinkingConfig(thinking_budget=-5).validate()

    # Coercion from int
    c_int = GeminiThinkingConfig.from_value(1024)
    assert c_int.thinking_budget == 1024
    assert c_int.enabled is True

    # Coercion from bool False -> thinking_budget = 0
    c_false = GeminiThinkingConfig.from_value(False)
    assert c_false.enabled is False
    assert c_false.thinking_budget == 0
    assert c_false.include_thoughts is False

    # Coercion from bool True -> thinking_budget = None (dynamic)
    c_true = GeminiThinkingConfig.from_value(True)
    assert c_true.enabled is True
    assert c_true.thinking_budget is None
    assert c_true.include_thoughts is True


# ==============================================================================
# OpenAIProvider Thinking Tests
# ==============================================================================


def test_openai_provider_thinking_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock(spec=["chat"])
    mock_choice = MagicMock()
    mock_choice.message.content = "Solved problem"
    mock_choice.message.tool_calls = None
    mock_usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        prompt_tokens_details=MagicMock(cached_tokens=10),
        completion_tokens_details=MagicMock(reasoning_tokens=150),
    )
    mock_resp = MagicMock(choices=[mock_choice], usage=mock_usage)
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="o3-mini", thinking="high")
    assert provider.thinking is not None
    assert provider.thinking.effort == "high"

    req = GenerationRequest(prompt="Math puzzle")
    res = provider.generate(req)

    assert res.text == "Solved problem"
    assert res.usage is not None
    assert res.usage.prompt_tokens == 100
    assert res.usage.completion_tokens == 200
    assert res.usage.total_tokens == 300
    assert res.usage.cached_tokens == 10
    assert res.usage.reasoning_tokens == 150

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "high"
    assert call_kwargs["model"] == "o3-mini"


def test_openai_provider_without_thinking_omits_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock(spec=["chat"])
    mock_choice = MagicMock()
    mock_choice.message.content = "Standard output"
    mock_choice.message.tool_calls = None
    mock_resp = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    # Standard model without thinking
    provider = OpenAIProvider(model="gpt-4o")
    assert provider.thinking is None
    res = provider.generate(GenerationRequest(prompt="Hello"))
    assert res.text == "Standard output"

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in call_kwargs


# ==============================================================================
# AnthropicProvider Thinking Tests
# ==============================================================================


def test_anthropic_provider_extended_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()

    mock_th_block = MagicMock(
        type="thinking",
        thinking="Step 1: Check constraints\nStep 2: Conclude",
        signature="sig_abc123",
    )
    mock_text_block = MagicMock(type="text", text="The answer is 42")
    mock_usage = MagicMock(
        input_tokens=150,
        output_tokens=350,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    mock_msg = MagicMock(content=[mock_th_block, mock_text_block], usage=mock_usage)
    mock_client.messages.create.return_value = mock_msg
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    # When budget_tokens is 2048 and max_tokens was 1024, auto-bump to 4096
    provider = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=2048)
    assert provider._max_tokens > 2048

    req = GenerationRequest(prompt="Deep question")
    res = provider.generate(req)

    assert res.text == "The answer is 42"
    assert res.thinking == "Step 1: Check constraints\nStep 2: Conclude"
    assert len(res.thought_blocks) == 1
    assert res.thought_blocks[0] == {
        "type": "thinking",
        "thinking": "Step 1: Check constraints\nStep 2: Conclude",
        "signature": "sig_abc123",
    }

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_anthropic_provider_multiturn_thought_signature_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(content=[MagicMock(type="text", text="Done")])
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-7-sonnet-20250219")

    # Replaying conversation history containing thought_blocks and tool_calls
    tc = ToolCall(id="call_99", name="fetch_data", arguments={"id": 1})
    thought_block = {
        "type": "thinking",
        "thinking": "Planning tool use...",
        "signature": "cryptographic_sig",
    }
    messages = [
        Message(role="user", content="Fetch data 1"),
        Message(role="assistant", content="", tool_calls=(tc,), thought_blocks=(thought_block,)),
        Message(role="tool", content="data result", tool_call_id="call_99", name="fetch_data"),
    ]
    req = GenerationRequest(messages=messages)
    provider.generate(req)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]
    assistant_msg = sent_messages[1]
    assert assistant_msg["role"] == "assistant"
    content_blocks = assistant_msg["content"]
    # Verify thought block is prepended with its signature intact
    assert content_blocks[0] == thought_block
    assert content_blocks[1]["type"] == "tool_use"
    assert content_blocks[1]["id"] == "call_99"


# ==============================================================================
# GeminiProvider Thinking Tests
# ==============================================================================


def test_gemini_provider_thinking_extraction_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    class FakeThinkingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakePart:
        @classmethod
        def from_text(cls, text: str) -> Any:
            p = MagicMock()
            p.text = text
            p.thought = False
            return p

    class FakeContent:
        def __init__(self, role: str, parts: list[Any]) -> None:
            self.role = role
            self.parts = parts

    mock_types.ThinkingConfig = FakeThinkingConfig  # type: ignore[attr-defined]
    mock_types.GenerateContentConfig = FakeGenerateContentConfig  # type: ignore[attr-defined]
    mock_types.Part = FakePart  # type: ignore[attr-defined]
    mock_types.Content = FakeContent  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_thought_part = MagicMock(text="Thinking through problem...", thought=True, function_call=None)
    mock_thought_part.thought_signature = None
    mock_thought_part.thoughtSignature = None
    mock_text_part = MagicMock(text="Final Gemini answer", thought=False, function_call=None)
    mock_text_part.thought_signature = None
    mock_text_part.thoughtSignature = None
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_thought_part, mock_text_part]

    mock_usage = MagicMock(
        prompt_token_count=50,
        candidates_token_count=100,
        total_token_count=150,
        cached_content_token_count=0,
        thinking_token_count=40,
        candidates_tokens_details=None,
    )
    mock_resp = MagicMock(candidates=[mock_candidate], usage_metadata=mock_usage)
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(
        model="gemini-2.5-flash",
        thinking=GeminiThinkingConfig(thinking_budget=2048, include_thoughts=True),
    )
    res = provider.generate(GenerationRequest(prompt="Explain logic"))

    assert res.text == "Final Gemini answer"
    assert res.thinking == "Thinking through problem..."
    assert len(res.thought_blocks) == 1
    assert res.thought_blocks[0] is mock_thought_part
    assert res.usage is not None
    assert res.usage.reasoning_tokens == 40

    call_config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert "thinking_config" in call_config.kwargs
    tc = call_config.kwargs["thinking_config"]
    assert tc.kwargs["thinking_budget"] == 2048
    assert tc.kwargs["include_thoughts"] is True


def test_gemini_provider_disable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    class FakeThinkingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    mock_types.ThinkingConfig = FakeThinkingConfig  # type: ignore[attr-defined]
    mock_types.GenerateContentConfig = FakeGenerateContentConfig  # type: ignore[attr-defined]
    mock_types.Part = MagicMock()  # type: ignore[attr-defined]
    mock_types.Content = MagicMock()  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(candidates=[MagicMock(content=MagicMock(parts=[]))])
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash", thinking=False)
    provider.generate(GenerationRequest(prompt="Fast query"))

    call_config = mock_client.models.generate_content.call_args.kwargs["config"]
    tc = call_config.kwargs["thinking_config"]
    assert tc.kwargs["thinking_budget"] == 0
    assert tc.kwargs["include_thoughts"] is False


# ==============================================================================
# Telemetry Visibility & Redaction Tests
# ==============================================================================


def test_llm_call_end_event_attributes_with_thinking() -> None:
    ctx = TraceContext.new_root()
    usage = TokenUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60, reasoning_tokens=30)
    event = LLMCallEndEvent(
        trace_context=ctx,
        provider="AnthropicProvider",
        model="claude-3-7-sonnet",
        output_text="Final text",
        thinking_text="Internal thoughts...",
        token_usage=usage,
    )

    attrs = event.to_attributes()
    assert attrs[semconv.GEN_AI_REASONING_CONTENT] == "Internal thoughts..."
    assert attrs[semconv.EVIDOR_THINKING_CONTENT] == "Internal thoughts..."
    assert attrs[semconv.GEN_AI_USAGE_REASONING_TOKENS] == 30


def test_redact_event_flag_control() -> None:
    ctx = TraceContext.new_root()
    event = LLMCallEndEvent(
        trace_context=ctx,
        provider="AnthropicProvider",
        model="claude-3-7-sonnet",
        output_text="Final text",
        thinking_text="Secret reasoning...",
    )

    # 1. capture_content=True, capture_thinking=True -> all retained
    e_all = redact_event(event, capture_content=True, capture_thinking=True)
    assert e_all.output_text == "Final text"
    assert e_all.thinking_text == "Secret reasoning..."

    # 2. capture_content=True, capture_thinking=False -> output retained, thinking stripped
    e_no_think = redact_event(event, capture_content=True, capture_thinking=False)
    assert e_no_think.output_text == "Final text"
    assert e_no_think.thinking_text is None
    attrs = e_no_think.to_attributes()
    assert semconv.GEN_AI_REASONING_CONTENT not in attrs

    # 3. capture_content=False, capture_thinking=False -> all content stripped
    e_none = redact_event(event, capture_content=False, capture_thinking=False)
    assert e_none.output_text == ""
    assert e_none.thinking_text is None


def test_in_memory_sink_capture_thinking_flag() -> None:
    ctx = TraceContext.new_root()
    event = LLMCallEndEvent(
        trace_context=ctx,
        provider="GeminiProvider",
        model="gemini-2.5-flash",
        output_text="Visible answer",
        thinking_text="Intermediate scratchpad",
    )

    # Sink with thinking disabled (default)
    sink_default = InMemorySink(capture_content=True, capture_thinking=False)
    sink_default.write([event])
    stored = sink_default.events[0]
    assert isinstance(stored, LLMCallEndEvent)
    assert stored.output_text == "Visible answer"
    assert stored.thinking_text is None

    # Sink with thinking enabled
    sink_thinking = InMemorySink(capture_content=True, capture_thinking=True)
    sink_thinking.write([event])
    stored_thinking = sink_thinking.events[0]
    assert isinstance(stored_thinking, LLMCallEndEvent)
    assert stored_thinking.output_text == "Visible answer"
    assert stored_thinking.thinking_text == "Intermediate scratchpad"


def test_agent_emits_thinking_telemetry() -> None:
    mock_provider = MagicMock()
    mock_resp = GenerationResponse(
        text="Agent answer",
        model="test-reasoner",
        thinking="Step-by-step agent deliberation",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=40, reasoning_tokens=25),
    )
    mock_provider.generate.return_value = mock_resp
    mock_provider.model = "test-reasoner"
    mock_provider.with_model.return_value = mock_provider

    sink = InMemorySink(capture_content=True, capture_thinking=True)
    agent = Agent(provider=mock_provider, telemetry=sink)
    res = agent.send("Do logic task")
    agent.close()

    assert res.text == "Agent answer"
    events = sink.filter(LLMCallEndEvent)
    assert len(events) == 1
    llm_end = events[0]
    assert llm_end.thinking_text == "Step-by-step agent deliberation"
    assert llm_end.token_usage is not None
    assert llm_end.token_usage.reasoning_tokens == 25

