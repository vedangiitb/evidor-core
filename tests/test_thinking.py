import sys
import types
from typing import Any
from unittest.mock import MagicMock
import pytest

from evidor import (
    Agent,
    AnthropicProvider,
    ConsoleSink,
    GeminiProvider,
    GenerationRequest,
    GenerationResponse,
    InMemorySink,
    Message,
    OpenAIProvider,
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


def test_thinking_config() -> None:
    # 1. Effort only
    cfg = ThinkingConfig(effort="high")
    assert cfg.effort == "high"
    assert cfg.budget_tokens is None
    assert cfg.summary is None
    cfg.validate()

    # 2. Budget tokens
    cfg_b = ThinkingConfig(budget_tokens=4096)
    assert cfg_b.budget_tokens == 4096

    # 3. Summary
    cfg_s = ThinkingConfig(summary="auto")
    assert cfg_s.summary == "auto"

    # 4. All combined
    cfg_all = ThinkingConfig(effort="xhigh", budget_tokens=2048, summary="concise")
    assert cfg_all.effort == "xhigh"
    assert cfg_all.budget_tokens == 2048
    assert cfg_all.summary == "concise"

    # 5. Coercion from shorthand string (Mode 1: thinking="high")
    c_str = ThinkingConfig.from_value("medium")
    assert c_str.effort == "medium"
    assert c_str.budget_tokens is None
    assert c_str.summary is None

    # 6. Coercion from dict
    c_dict = ThinkingConfig.from_value({"effort": "low", "summary": "none"})
    assert c_dict.effort == "low"
    assert c_dict.summary == "none"

    # 7. Identity
    assert ThinkingConfig.from_value(cfg) is cfg

    # 8. No enabled parameter exists
    with pytest.raises(TypeError):
        ThinkingConfig(enabled=True)  # type: ignore[call-arg]

    # 9. No integer shorthand (thinking=4096 is confusing, must use ThinkingConfig(budget_tokens=4096))
    with pytest.raises(TypeError, match="Cannot coerce int to ThinkingConfig"):
        ThinkingConfig.from_value(4096)

    # 10. No boolean shorthand (thinking=False not supported)
    with pytest.raises(TypeError, match="Cannot coerce bool to ThinkingConfig"):
        ThinkingConfig.from_value(False)


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
    assert provider.thinking == "high"

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
    provider = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=ThinkingConfig(budget_tokens=2048))
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


def test_anthropic_builder_adaptive_effort_and_summary() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    # 1. Adaptive with effort and summary="auto" -> display: "summarized"
    cfg1 = ThinkingConfig(effort="high", summary="auto")
    b1 = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=2048, thinking=cfg1)
    p1 = b1.build(GenerationRequest(prompt="Explain quantum mechanics"))
    assert p1["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert p1["output_config"] == {"effort": "high"}

    # 2. Adaptive with effort and summary="none" -> display: "omitted"
    cfg2 = ThinkingConfig(effort="low", summary="none")
    b2 = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=2048, thinking=cfg2)
    p2 = b2.build(GenerationRequest(prompt="Simple math"))
    assert p2["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert p2["output_config"] == {"effort": "low"}

    # 3. Summary only ("detailed") -> adaptive thinking with display: "summarized"
    cfg3 = ThinkingConfig(summary="detailed")
    b3 = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=2048, thinking=cfg3)
    p3 = b3.build(GenerationRequest(prompt="Summarize"))
    assert p3["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "output_config" not in p3

    # 4. Summary only ("omitted") -> adaptive thinking with display: "omitted"
    cfg4 = ThinkingConfig(summary="omitted")
    b4 = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=2048, thinking=cfg4)
    p4 = b4.build(GenerationRequest(prompt="Code quickly"))
    assert p4["thinking"] == {"type": "adaptive", "display": "omitted"}

    # 5. Budget tokens with summary="concise" -> display: "summarized"
    cfg5 = ThinkingConfig(budget_tokens=2048, summary="concise")
    b5 = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=2048, thinking=cfg5)
    p5 = b5.build(GenerationRequest(prompt="Analyze code"))
    assert p5["thinking"] == {"type": "enabled", "budget_tokens": 2048, "display": "summarized"}
    assert p5["max_tokens"] > 2048

    # 6. Budget tokens with summary="none" -> display: "omitted"
    cfg6 = ThinkingConfig(budget_tokens=2048, summary="none")
    b6 = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=4096, thinking=cfg6)
    p6 = b6.build(GenerationRequest(prompt="Deep benchmark"))
    assert p6["thinking"] == {"type": "enabled", "budget_tokens": 2048, "display": "omitted"}


def test_anthropic_builder_thought_block_normalization_cross_provider() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet", max_tokens=2048)

    # Gemini-style thought block with thought_signature and boolean thought flag
    gem_block = {
        "type": "thought",
        "text": "Gemini chain-of-thought",
        "thought": True,
        "thought_signature": "gem_sig_789",
        "extra_gemini_field": "ignore_me",
    }
    norm_gem = builder._normalize_thought_block(gem_block)
    assert norm_gem == {
        "type": "thinking",
        "thinking": "Gemini chain-of-thought",
        "signature": "gem_sig_789",
    }
    assert "extra_gemini_field" not in norm_gem
    assert "thought" not in norm_gem

    # OpenAI-style reasoning block with summary
    oai_block = {
        "type": "reasoning",
        "summary": "OpenAI reasoning summary",
        "signature": "oai_sig_321",
    }
    norm_oai = builder._normalize_thought_block(oai_block)
    assert norm_oai == {
        "type": "thinking",
        "thinking": "OpenAI reasoning summary",
        "signature": "oai_sig_321",
    }

    # Anthropic omitted block (empty thinking)
    omitted_block = {
        "type": "thinking",
        "thinking": "",
        "signature": "omitted_sig_999",
    }
    norm_omit = builder._normalize_thought_block(omitted_block)
    assert norm_omit == {
        "type": "thinking",
        "thinking": "",
        "signature": "omitted_sig_999",
    }

    # Redacted thinking block
    redacted_block = {
        "type": "redacted_thinking",
        "data": "encrypted_thought_data",
    }
    norm_redacted = builder._normalize_thought_block(redacted_block)
    assert norm_redacted == {
        "type": "redacted_thinking",
        "data": "encrypted_thought_data",
    }

    # Replay assistant message with normalized blocks and fallback for empty content
    req = GenerationRequest(
        messages=[
            Message(role="user", content="Test"),
            Message(role="assistant", content="", thought_blocks=(gem_block,)),
        ]
    )
    payload = builder.build(req)
    asst_content = payload["messages"][1]["content"]
    assert len(asst_content) == 1
    assert asst_content[0]["type"] == "thinking"
    assert asst_content[0]["signature"] == "gem_sig_789"


def test_anthropic_parser_reasoning_summary_and_omitted() -> None:
    from evidor.agent.providers.anthropic.parser import AnthropicResponseParser

    parser = AnthropicResponseParser(model="claude-3-7-sonnet")

    # 1. Summarized thinking block: thinking has text
    resp_sum = {
        "id": "msg_sum_01",
        "content": [
            {"type": "thinking", "summary": "Core algorithm design steps", "signature": "sig_sum_1"},
            {"type": "text", "text": "Here is the implementation"},
        ],
    }
    res_sum = parser.parse(resp_sum)
    assert res_sum.thinking == "Core algorithm design steps"
    assert len(res_sum.thought_blocks) == 1
    assert res_sum.thought_blocks[0] == {
        "type": "thinking",
        "thinking": "Core algorithm design steps",
        "signature": "sig_sum_1",
    }

    # 2. Omitted thinking block: thinking is empty string ""
    resp_omit = {
        "id": "msg_omit_02",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig_omit_2"},
            {"type": "text", "text": "Fast response"},
        ],
    }
    res_omit = parser.parse(resp_omit)
    assert res_omit.thinking is None  # Omitted thinking text should not expose empty string
    assert len(res_omit.thought_blocks) == 1
    assert res_omit.thought_blocks[0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig_omit_2",
    }


def test_anthropic_stream_reasoning_summary_and_signature_accumulation() -> None:
    from evidor.agent.providers.anthropic.stream import AnthropicStreamHandler

    handler = AnthropicStreamHandler(model="claude-3-7-sonnet")

    events = [
        {"type": "message_start", "message": {"id": "msg_str_01", "model": "claude-3-7-sonnet"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "First reasoning step. "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_summary_delta", "summary": "Second reasoning step."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig_part1_"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig_part2"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Final answer."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    chunks = []
    for ev in events:
        chunks.extend(list(handler.process_event(ev)))

    finished_chunk = next(c for c in chunks if c.is_finished)
    final_resp = finished_chunk.response
    assert final_resp is not None
    assert final_resp.text == "Final answer."
    assert final_resp.thinking == "First reasoning step. Second reasoning step."
    assert len(final_resp.thought_blocks) == 1
    assert final_resp.thought_blocks[0] == {
        "type": "thinking",
        "thinking": "First reasoning step. Second reasoning step.",
        "signature": "sig_part1_sig_part2",
    }


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
        thinking=ThinkingConfig(effort="medium", summary="auto"),
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

    # Test Gemini 3.x thinking_level parameter (Mode 1: thinking="high")
    p_gemini_3 = GeminiProvider(model="gemini-3.8-flash", thinking="high")
    assert p_gemini_3.thinking == "high"
    res_3 = p_gemini_3.generate(GenerationRequest(prompt="Explain logic"))
    call_config_3 = mock_client.interactions.create.call_args.kwargs["generation_config"]
    assert call_config_3["thinking_level"] == "high"
    assert "thinking_summaries" not in call_config_3

    # Test Mode 2 with thinking_budget (earlier models)
    p_gemini_budget = GeminiProvider(model="gemini-2.5-flash", thinking=ThinkingConfig(budget_tokens=2048))
    assert p_gemini_budget.thinking.budget_tokens == 2048
    p_gemini_budget.generate(GenerationRequest(prompt="Explain logic"))
    call_config_budget = mock_client.interactions.create.call_args.kwargs["generation_config"]
    assert call_config_budget["thinking_budget"] == 2048


def test_gemini_provider_without_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MagicMock(output_text="Fast answer", steps=[], usage=None)
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    provider.generate(GenerationRequest(prompt="Fast query"))

    # When thinking is omitted or None, no generation_config is passed
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert "generation_config" not in call_kwargs


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
    assert p1.thinking == "high"

    p2 = OpenAIProvider(model="gpt-4o", thinking="low")
    assert p2.thinking == "low"

    p3 = OpenAIProvider(model="gpt-6-astra", thinking="xhigh")
    assert p3.thinking == "xhigh"

    p_sol = OpenAIProvider(model="gpt-6.1-sol", thinking="medium")
    assert p_sol.thinking == "medium"

    p_luna = OpenAIProvider(model="gpt-6-luna", thinking="none")
    assert p_luna.thinking == "none"

    p4 = OpenAIProvider(model="gpt-6-astra", thinking=ThinkingConfig(effort="max"))
    assert p4.thinking.effort == "max"


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
        thinking=ThinkingConfig(effort="low", summary="auto"),
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
    p_any = AnthropicProvider(model="claude-future-model", thinking=ThinkingConfig(budget_tokens=2048))
    assert p_any.thinking.budget_tokens == 2048
    assert p_any._max_tokens == 2048 + 2048

    # When user explicitly provides max_tokens, it is respected as-is without client-side blocking
    p_explicit = AnthropicProvider(model="claude-future-model", max_tokens=1500, thinking=ThinkingConfig(budget_tokens=2048))
    assert p_explicit._max_tokens == 1500

    # Any future effort level string is passed through
    p_older = AnthropicProvider(model="claude-3-5-sonnet", thinking="ultra")
    assert p_older.thinking == "ultra"

    p2 = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=ThinkingConfig(effort="high"))
    assert p2.thinking.effort == "high"


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



