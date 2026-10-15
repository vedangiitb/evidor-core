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

    # Any custom or newly released effort mode is accepted
    c_future = OpenAIThinkingConfig.from_value("ultra-high")
    assert c_future.effort == "ultra-high"
    assert c_future.enabled is True

    # Invalid type
    with pytest.raises(TypeError, match="Cannot coerce"):
        OpenAIThinkingConfig.from_value(1234)


def test_anthropic_thinking_config() -> None:
    cfg = AnthropicThinkingConfig(budget_tokens=4096)
    assert cfg.budget_tokens == 4096
    assert cfg.type == "enabled"
    cfg.validate()

    # Adaptive effort
    adaptive = AnthropicThinkingConfig(type="adaptive", effort="high", budget_tokens=None)
    adaptive.validate()
    assert adaptive.type == "adaptive"
    assert adaptive.effort == "high"

    # Coercion from integer budget
    c_int = AnthropicThinkingConfig.from_value(2048)
    assert c_int.budget_tokens == 2048
    assert c_int.type == "enabled"

    # Coercion from effort string (including any future effort level)
    c_str = AnthropicThinkingConfig.from_value("ultra")
    assert c_str.type == "adaptive"
    assert c_str.effort == "ultra"

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
    assert res.thought_blocks[0]["text"] == "Thinking through problem..."
    assert res.thought_blocks[0]["thought"] is True
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


def test_redact_event_capture_content_false_capture_thinking_true() -> None:
    """Security boundary test: capture_thinking=True cannot override capture_content=False."""
    ctx = TraceContext.new_root()
    usage = TokenUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60, reasoning_tokens=35)
    event = LLMCallEndEvent(
        trace_context=ctx,
        provider="AnthropicProvider",
        model="claude-3-7-sonnet",
        output_text="Secret final answer",
        thinking_text="Private internal deliberation",
        token_usage=usage,
    )

    redacted = redact_event(event, capture_content=False, capture_thinking=True)

    # Text content must be stripped
    assert redacted.output_text == ""
    assert redacted.thinking_text is None

    # Quantitative token counts must be preserved
    assert redacted.token_usage is not None
    assert redacted.token_usage.reasoning_tokens == 35

    attrs = redacted.to_attributes()
    assert semconv.GEN_AI_REASONING_CONTENT not in attrs
    assert semconv.EVIDOR_THINKING_CONTENT not in attrs
    assert attrs[semconv.GEN_AI_USAGE_REASONING_TOKENS] == 35


def test_openai_thinking_forward_compatible_models() -> None:
    # Any model name is accepted without client-side gating for forward compatibility
    p1 = OpenAIProvider(model="o3-mini", thinking="high")
    assert p1.thinking.effort == "high"

    p2 = OpenAIProvider(model="gpt-4o", thinking="low")
    assert p2.thinking.effort == "low"

    p3 = OpenAIProvider(model="future-o4-preview", thinking="medium")
    assert p3.thinking.effort == "medium"

    p4 = OpenAIProvider(model="gpt-4o", thinking=False)
    assert p4.thinking.enabled is False


def test_openai_responses_api_thinking_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()

    class FakeReasoningSummaryText:
        def __init__(self, text: str, type: str = "text") -> None:
            self.text = text
            self.type = type

    # Responses API reasoning output returns summary as a list of structured objects
    mock_summary_part = FakeReasoningSummaryText(text="Responses API deliberate steps")
    mock_reasoning_item = MagicMock(
        type="reasoning",
        summary=[mock_summary_part],
        content=[],
    )
    mock_text_item = MagicMock(type="message")
    mock_resp = MagicMock(
        output=[mock_reasoning_item, mock_text_item],
        output_text="Responses API final answer",
        usage=MagicMock(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            prompt_tokens_details=None,
            completion_tokens_details=MagicMock(reasoning_tokens=80),
        ),
    )
    mock_client.responses.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="o3-mini", thinking="high")
    res = provider.generate(GenerationRequest(prompt="Deep math"))

    assert res.text == "Responses API final answer"
    assert res.thinking == "Responses API deliberate steps"
    assert res.usage is not None
    assert res.usage.reasoning_tokens == 80

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in call_kwargs


def test_openai_responses_api_with_summary_config(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_resp = MagicMock(
        output=[],
        output_text="Done",
        usage=None,
    )
    mock_client.responses.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(
        model="o3-mini",
        thinking=OpenAIThinkingConfig(effort="low", summary="auto"),
    )
    provider.generate(GenerationRequest(prompt="Summarized steps"))

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "reasoning_effort" not in call_kwargs


def test_openai_chat_completions_reasoning_content_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock(spec=["chat"])
    mock_choice = MagicMock()
    mock_choice.message.content = "Final solution"
    mock_choice.message.reasoning_content = "DeepSeek-R1 style reasoning steps"
    mock_choice.message.tool_calls = None
    mock_usage = MagicMock(
        prompt_tokens=40,
        completion_tokens=120,
        total_tokens=160,
        prompt_tokens_details=None,
        completion_tokens_details=MagicMock(reasoning_tokens=90),
    )
    mock_resp = MagicMock(choices=[mock_choice], usage=mock_usage)
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="o1", thinking="medium")
    res = provider.generate(GenerationRequest(prompt="Logic puzzle"))

    assert res.text == "Final solution"
    assert res.thinking == "DeepSeek-R1 style reasoning steps"
    assert res.usage is not None
    assert res.usage.reasoning_tokens == 90


def test_anthropic_forward_compatibility() -> None:
    # Any model name is accepted without client-side gating for forward compatibility
    p_any = AnthropicProvider(model="claude-future-model", thinking=2048)
    assert p_any.thinking.budget_tokens == 2048
    assert p_any._max_tokens == 2048 + 2048

    # When user explicitly provides max_tokens, it is respected as-is without client-side blocking
    p_explicit = AnthropicProvider(model="claude-future-model", max_tokens=1500, thinking=2048)
    assert p_explicit._max_tokens == 1500

    # Any future effort level string is passed through
    p_older = AnthropicProvider(model="claude-3-5-sonnet", thinking="ultra")
    assert p_older.thinking.effort == "ultra"

    p2 = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=False)
    assert p2.thinking.enabled is False


def test_gemini_multiturn_thought_signature_preservation(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    class FakePart:
        def __init__(self, text: str = "", thought: bool = False, thought_signature: Any = None, **kwargs: Any) -> None:
            self.text = text
            self.thought = thought
            self.thought_signature = thought_signature
            self.function_call = None
            self.kwargs = kwargs

        @classmethod
        def from_text(cls, text: str) -> Any:
            return cls(text=text)

        @classmethod
        def from_function_call(cls, name: str, args: dict[str, Any]) -> Any:
            p = cls()
            fc = MagicMock()
            fc.name = name
            fc.args = args
            p.function_call = fc
            return p

        @classmethod
        def from_function_response(cls, name: str, response: dict[str, Any]) -> Any:
            p = cls()
            fr = MagicMock()
            fr.name = name
            fr.response = response
            p.function_response = fr
            return p

    class FakeContent:
        def __init__(self, role: str, parts: list[Any]) -> None:
            self.role = role
            self.parts = parts

    mock_types.Part = FakePart  # type: ignore[attr-defined]
    mock_types.Content = FakeContent  # type: ignore[attr-defined]
    mock_types.GenerateContentConfig = MagicMock()  # type: ignore[attr-defined]
    mock_types.ThinkingConfig = MagicMock()  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[]))],
        usage_metadata=None,
    )
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash")

    # Multi-turn conversation with thought block containing cryptographic signature
    tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
    thought_block = {
        "text": "Internal Gemini reasoning...",
        "thought": True,
        "thought_signature": b"crypto_gemini_sig_xyz",
    }
    messages = [
        Message(role="user", content="Search query"),
        Message(role="assistant", content="", tool_calls=(tc,), thought_blocks=(thought_block,)),
        Message(role="tool", content="search result", tool_call_id="call_1", name="search"),
    ]

    provider.generate(GenerationRequest(messages=messages))

    call_contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assistant_content = call_contents[1]
    assert assistant_content.role == "model"

    # Verify that the thought part was reconstructed with thought=True and signature intact
    th_part = assistant_content.parts[0]
    assert th_part.thought is True
    assert th_part.text == "Internal Gemini reasoning..."
    assert th_part.thought_signature == b"crypto_gemini_sig_xyz"


def test_gemini_provider_function_call_thought_signature_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    class FakeFunctionCall:
        def __init__(self, name: str, args: dict[str, Any]) -> None:
            self.name = name
            self.args = args

    class FakePart:
        def __init__(
            self,
            text: str | None = None,
            thought: bool | None = None,
            thought_signature: Any = None,
            function_call: Any = None,
        ) -> None:
            self.text = text
            self.thought = thought
            self.thought_signature = thought_signature
            self.function_call = function_call

        @classmethod
        def from_text(cls, text: str) -> "FakePart":
            return cls(text=text)

        @classmethod
        def from_function_call(cls, name: str, args: dict[str, Any]) -> "FakePart":
            return cls(function_call=FakeFunctionCall(name=name, args=args))

        @classmethod
        def from_function_response(cls, name: str, response: dict[str, Any]) -> "FakePart":
            p = cls()
            setattr(p, "function_response", {"name": name, "response": response})
            return p

    class FakeContent:
        def __init__(self, role: str, parts: list[Any]) -> None:
            self.role = role
            self.parts = parts

    mock_types.Part = FakePart  # type: ignore[attr-defined]
    mock_types.Content = FakeContent  # type: ignore[attr-defined]
    mock_types.GenerateContentConfig = MagicMock()  # type: ignore[attr-defined]
    mock_types.ThinkingConfig = MagicMock()  # type: ignore[attr-defined]

    mock_client = MagicMock()
    # Turn 1: Model response has a function call part with thought_signature directly attached to it
    fc_part = FakePart.from_function_call(name="execute_code", args={"code": "print(42)"})
    fc_part.thought_signature = b"gemini_fc_sig_binary_999"

    mock_resp_1 = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[fc_part]))],
        usage_metadata=None,
    )
    # Turn 2: Final response text
    final_text_part = FakePart.from_text("Output: 42")
    mock_resp_2 = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[final_text_part]))],
        usage_metadata=None,
    )
    mock_client.models.generate_content.side_effect = [mock_resp_1, mock_resp_2, mock_resp_2]
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash")

    # Turn 1 execution
    res_1 = provider.generate(GenerationRequest(prompt="Run code"))
    assert len(res_1.tool_calls) == 1
    assert res_1.tool_calls[0].name == "execute_code"
    assert res_1.tool_calls[0].arguments == {"code": "print(42)"}
    assert len(res_1.thought_blocks) == 1
    assert res_1.thought_blocks[0]["type"] == "function_call"
    assert res_1.thought_blocks[0]["thought_signature"] == b"gemini_fc_sig_binary_999"

    # Turn 2 multi-turn tool replay: assistant message has tool_calls and thought_blocks
    messages = [
        Message(role="user", content="Run code"),
        Message(role="assistant", content="", tool_calls=res_1.tool_calls, thought_blocks=res_1.thought_blocks),
        Message(role="tool", content="42", tool_call_id=res_1.tool_calls[0].id, name="execute_code"),
    ]
    res_2 = provider.generate(GenerationRequest(messages=messages))
    assert res_2.text == "Output: 42"

    second_call_contents = mock_client.models.generate_content.call_args_list[1].kwargs["contents"]
    assistant_content = second_call_contents[1]
    assert assistant_content.role == "model"
    # Ensure thought signature is attached directly to the function-call part
    assert len(assistant_content.parts) == 1
    replayed_part = assistant_content.parts[0]
    assert replayed_part.function_call.name == "execute_code"
    assert replayed_part.function_call.args == {"code": "print(42)"}
    assert replayed_part.thought_signature == b"gemini_fc_sig_binary_999"
    # Ensure it is NOT converted into a thought text part
    assert getattr(replayed_part, "thought", None) is not True

    # Also verify replay with a deserialized dictionary representation (e.g. from JSON storage without _raw_part)
    deserialized_fc_block = {
        "type": "function_call",
        "name": "execute_code",
        "args": {"code": "print(42)"},
        "thought_signature": b"gemini_fc_sig_binary_999",
    }
    messages_deserialized = [
        Message(role="user", content="Run code"),
        Message(role="assistant", content="", tool_calls=res_1.tool_calls, thought_blocks=(deserialized_fc_block,)),
        Message(role="tool", content="42", tool_call_id=res_1.tool_calls[0].id, name="execute_code"),
    ]
    provider.generate(GenerationRequest(messages=messages_deserialized))

    replayed_contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    replayed_assistant = replayed_contents[1]
    assert len(replayed_assistant.parts) == 1
    reconstructed_part = replayed_assistant.parts[0]
    assert reconstructed_part.function_call.name == "execute_code"
    assert reconstructed_part.thought_signature == b"gemini_fc_sig_binary_999"


