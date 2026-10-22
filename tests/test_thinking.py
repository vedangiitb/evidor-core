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
    cfg = GeminiThinkingConfig(thinking_level="high", thinking_summaries="auto")
    assert cfg.thinking_level == "high"
    assert cfg.thinking_summaries == "auto"
    cfg.validate()

    # Dynamic / None
    cfg_dyn = GeminiThinkingConfig(thinking_level=None)
    cfg_dyn.validate()

    # Disabled
    cfg_off = GeminiThinkingConfig(enabled=False)
    cfg_off.validate()

    # Coercion from int -> raises TypeError because Interactions API does not support numeric budgets
    with pytest.raises(TypeError, match="Numeric budgets are not supported"):
        GeminiThinkingConfig.from_value(1024)

    # Coercion from bool False -> minimal + none
    c_false = GeminiThinkingConfig.from_value(False)
    assert c_false.enabled is False
    assert c_false.thinking_level == "minimal"
    assert c_false.thinking_summaries == "none"

    # Coercion from bool True
    c_true = GeminiThinkingConfig.from_value(True)
    assert c_true.enabled is True

    # Coercion from str -> thinking_level (Gemini 3.x)
    c_str = GeminiThinkingConfig.from_value("high")
    assert c_str.enabled is True
    assert c_str.thinking_level == "high"


# ==============================================================================
# OpenAIProvider Thinking Tests
# ==============================================================================


def test_openai_provider_thinking_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        prompt_tokens_details=MagicMock(cached_tokens=10),
        completion_tokens_details=MagicMock(reasoning_tokens=150),
    )
    mock_resp = MagicMock(
        output_text="Solved problem",
        output=[MagicMock(type="reasoning", summary="Step-by-step logic", text="Step-by-step logic")],
        usage=mock_usage,
        id="resp_think",
    )
    mock_client.responses.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", thinking="high")
    assert provider.thinking is not None
    assert provider.thinking.effort == "high"

    req = GenerationRequest(prompt="Math puzzle")
    res = provider.generate(req)

    assert res.text == "Solved problem"
    assert res.thinking == "Step-by-step logic"
    assert res.usage is not None
    assert res.usage.prompt_tokens == 100
    assert res.usage.completion_tokens == 200
    assert res.usage.total_tokens == 300
    assert res.usage.cached_tokens == 10
    assert res.usage.reasoning_tokens == 150

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["reasoning"] == {"effort": "high"}
    assert call_kwargs["model"] == "gpt-6-astra"


def test_openai_provider_without_thinking_omits_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_resp = MagicMock(output_text="Standard output", output=[], usage=None, id="resp_no_think")
    mock_client.responses.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    # Standard model without thinking
    provider = OpenAIProvider(model="gpt-4.1")
    assert provider.thinking is None
    res = provider.generate(GenerationRequest(prompt="Hello"))
    assert res.text == "Standard output"

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert "reasoning" not in call_kwargs


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

    mock_client = MagicMock()
    mock_thought_step = MagicMock(
        type="thought",
        signature=None,
        summary=[MagicMock(text="Thinking through problem...")],
    )
    mock_output_step = MagicMock(
        type="model_output",
        content=[MagicMock(text="Final Gemini answer")],
    )

    mock_usage = MagicMock(
        total_input_tokens=50,
        total_output_tokens=100,
        total_tokens=150,
        total_cached_tokens=0,
        total_thought_tokens=40,
    )
    mock_resp = MagicMock(
        output_text="Final Gemini answer",
        steps=[mock_thought_step, mock_output_step],
        usage=mock_usage,
    )
    mock_client.interactions.create.return_value = mock_resp
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(
        model="gemini-3.8-flash",
        thinking=GeminiThinkingConfig(thinking_level="medium", thinking_summaries="auto"),
    )
    res = provider.generate(GenerationRequest(prompt="Explain logic"))

    assert res.text == "Final Gemini answer"
    assert res.thinking == "Thinking through problem..."
    assert len(res.thought_blocks) == 1
    assert res.thought_blocks[0]["text"] == "Thinking through problem..."
    assert res.thought_blocks[0]["thought"] is True
    assert res.usage is not None
    assert res.usage.reasoning_tokens == 40

    call_config = mock_client.interactions.create.call_args.kwargs["generation_config"]
    assert call_config["thinking_level"] == "medium"
    assert call_config["thinking_summaries"] == "auto"

    # Test Gemini 3.x thinking_level parameter
    p_gemini_3 = GeminiProvider(model="gemini-3.8-flash", thinking="high")
    assert p_gemini_3.thinking.thinking_level == "high"
    res_3 = p_gemini_3.generate(GenerationRequest(prompt="Explain logic"))
    call_config_3 = mock_client.interactions.create.call_args.kwargs["generation_config"]
    assert call_config_3["thinking_level"] == "high"
    assert call_config_3["thinking_summaries"] == "auto"


def test_gemini_provider_disable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MagicMock(output_text="Fast answer", steps=[], usage=None)
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", thinking=False)
    provider.generate(GenerationRequest(prompt="Fast query"))

    call_config = mock_client.interactions.create.call_args.kwargs["generation_config"]
    assert call_config["thinking_level"] == "minimal"


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

    p3 = OpenAIProvider(model="gpt-6-astra", thinking="xhigh")
    assert p3.thinking.effort == "xhigh"

    p_sol = OpenAIProvider(model="gpt-6.1-sol", thinking="medium")
    assert p_sol.thinking.effort == "medium"

    p_luna = OpenAIProvider(model="gpt-6-luna", thinking="none")
    assert p_luna.thinking.effort == "none"

    p4 = OpenAIProvider(model="gpt-6-astra", thinking=False)
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


def test_openai_responses_api_modern_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_usage = MagicMock(
        spec=["input_tokens", "output_tokens", "total_tokens", "input_tokens_details", "output_tokens_details"],
        input_tokens=120,
        output_tokens=250,
        total_tokens=370,
        input_tokens_details=MagicMock(cached_tokens=40),
        output_tokens_details=MagicMock(reasoning_tokens=180),
    )
    mock_resp = MagicMock(
        output=[],
        output_text="Modern response",
        usage=mock_usage,
    )
    mock_client.responses.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", thinking="xhigh")
    res = provider.generate(GenerationRequest(prompt="Astra prompt"))

    assert res.text == "Modern response"
    assert res.usage is not None
    assert res.usage.prompt_tokens == 120
    assert res.usage.completion_tokens == 250
    assert res.usage.total_tokens == 370
    assert res.usage.cached_tokens == 40
    assert res.usage.reasoning_tokens == 180


def test_openai_chat_completions_reasoning_content_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_usage = MagicMock(
        prompt_tokens=40,
        completion_tokens=120,
        total_tokens=160,
        prompt_tokens_details=None,
        completion_tokens_details=MagicMock(reasoning_tokens=90),
    )
    mock_reasoning_item = MagicMock(type="reasoning", summary=None, text="DeepSeek-R1 style reasoning steps")
    mock_resp = MagicMock(
        output_text="Final solution",
        output=[mock_reasoning_item],
        usage=mock_usage,
        id="resp_extract",
    )
    mock_client.responses.create.return_value = mock_resp
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", thinking="medium")
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

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MagicMock(
        output_text="Search result summary",
        steps=[],
        usage=None,
    )
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")

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

    call_input = mock_client.interactions.create.call_args.kwargs["input"]
    assert len(call_input) == 4
    th_step = call_input[1]
    assert th_step["type"] == "thought"
    assert th_step["signature"] == b"crypto_gemini_sig_xyz"
    assert th_step["summary"] == [{"type": "text", "text": "Internal Gemini reasoning..."}]

    fc_step = call_input[2]
    assert fc_step["type"] == "function_call"
    assert fc_step["id"] == "call_1"
    assert fc_step["name"] == "search"
    assert fc_step["arguments"] == {"q": "test"}

    fr_step = call_input[3]
    assert fr_step["type"] == "function_result"
    assert fr_step["call_id"] == "call_1"
    assert fr_step["name"] == "search"
    assert fr_step["result"] == {"result": "search result"}


def test_gemini_provider_function_call_thought_signature_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()
    # Turn 1: Model response has a function call step with signature
    mock_fc_step = MagicMock(
        type="function_call",
        id="call_fc_1",
        arguments={"code": "print(42)"},
        signature=b"gemini_fc_sig_binary_999",
    )
    mock_fc_step.name = "execute_code"

    mock_resp_1 = MagicMock(
        output_text="",
        steps=[mock_fc_step],
        usage=None,
    )
    # Turn 2: Final response text
    mock_output_step = MagicMock(
        type="model_output",
        content=[MagicMock(text="Output: 42")],
    )
    mock_resp_2 = MagicMock(
        output_text="Output: 42",
        steps=[mock_output_step],
        usage=None,
    )
    mock_client.interactions.create.side_effect = [mock_resp_1, mock_resp_2, mock_resp_2]
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")

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

    second_call_input = mock_client.interactions.create.call_args_list[1].kwargs["input"]
    # Ensure thought signature is attached directly to the replayed function-call step
    fc_replayed = second_call_input[1]
    assert fc_replayed["type"] == "function_call"
    assert fc_replayed["name"] == "execute_code"
    assert fc_replayed["arguments"] == {"code": "print(42)"}
    assert fc_replayed["signature"] == b"gemini_fc_sig_binary_999"

    # Also verify replay with a deserialized dictionary representation (e.g. from JSON storage without _raw_part)
    deserialized_fc_block = {
        "type": "function_call",
        "id": res_1.tool_calls[0].id,
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

    replayed_input = mock_client.interactions.create.call_args.kwargs["input"]
    reconstructed_step = replayed_input[1]
    assert reconstructed_step["type"] == "function_call"
    assert reconstructed_step["name"] == "execute_code"
    assert reconstructed_step["signature"] == b"gemini_fc_sig_binary_999"



