import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from evidor import (
    AnthropicProvider,
    GeminiProvider,
    GenerationRequest,
    GenerationResponse,
    Message,
    OpenAIProvider,
    Tool,
    ToolCall,
)


# --- OpenAIProvider Tests ---


def test_openai_provider_properties_and_with_model() -> None:
    provider = OpenAIProvider(model="gpt-4.1", api_key="sk-test")
    assert provider.model == "gpt-4.1"

    new_provider = provider.with_model("gpt-4.1-mini")
    assert new_provider.model == "gpt-4.1-mini"
    assert new_provider._api_key == "sk-test"


def test_openai_provider_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_response = MagicMock(output_text="OpenAI generated text")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-4.1-mini", api_key="key")
    request = GenerationRequest(
        messages=[
            Message(role="system", content="sys instruction"),
            Message(role="user", content="hello"),
        ]
    )

    response = provider.generate(request)

    assert response == GenerationResponse(text="OpenAI generated text", model="gpt-4.1-mini")
    mock_openai.OpenAI.assert_called_once_with(api_key="key")
    mock_client.responses.create.assert_called_once_with(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "sys instruction"},
            {"role": "user", "content": "hello"},
        ],
    )




def test_openai_provider_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = OpenAIProvider(model="gpt-4.1-mini")
    with pytest.raises(ImportError, match="Install OpenAI support with: pip install 'evidor\\[openai\\]'"):
        provider.generate(GenerationRequest(prompt="test"))


# --- AnthropicProvider Tests ---


def test_anthropic_provider_properties_and_with_model() -> None:
    provider = AnthropicProvider(model="claude-3-5", api_key="ant-key", max_tokens=2048)
    assert provider.model == "claude-3-5"
    assert provider.enable_prompt_caching is True

    new_provider = provider.with_model("claude-3-haiku")
    assert new_provider.model == "claude-3-haiku"
    assert new_provider._api_key == "ant-key"
    assert new_provider._max_tokens == 2048
    assert new_provider.enable_prompt_caching is True


def test_anthropic_provider_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_block1 = MagicMock(text="Claude ", type="text")
    mock_block2 = MagicMock(text="response", type="text")
    mock_block_other = MagicMock(type="other")
    mock_message = MagicMock(content=[mock_block1, mock_block2, mock_block_other])
    mock_client.messages.create.return_value = mock_message
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-5", api_key="key", max_tokens=500)
    request = GenerationRequest(
        messages=[
            Message(role="system", content="sys instruction"),
            Message(role="user", content="hello"),
        ]
    )

    response = provider.generate(request)

    assert response == GenerationResponse(text="Claude response", model="claude-3-5")
    mock_anthropic.Anthropic.assert_called_once_with(api_key="key")
    mock_client.messages.create.assert_called_once_with(
        model="claude-3-5",
        max_tokens=500,
        system="sys instruction",
        messages=[{"role": "user", "content": "hello"}],
        cache_control={"type": "ephemeral"},
    )




def test_anthropic_provider_generate_no_system(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(content=[MagicMock(text="ok", type="text")])
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-5")
    provider.generate(GenerationRequest(prompt="hello"))
    assert mock_client.messages.create.call_args.kwargs["system"] is None


def test_anthropic_provider_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = AnthropicProvider(model="claude-3-5")
    with pytest.raises(ImportError, match="Install Anthropic support with: pip install 'evidor\\[anthropic\\]'"):
        provider.generate(GenerationRequest(prompt="test"))


# --- GeminiProvider Tests ---


def test_gemini_provider_properties_and_with_model() -> None:
    provider = GeminiProvider(model="gemini-2.5-flash", api_key="gem-key")
    assert provider.model == "gemini-2.5-flash"

    new_provider = provider.with_model("gemini-2.5-pro")
    assert new_provider.model == "gemini-2.5-pro"
    assert new_provider._api_key == "gem-key"


def test_gemini_provider_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_types.HttpOptions = MagicMock(side_effect=lambda **kw: {"http_options": kw})  # type: ignore[attr-defined]
    mock_types.HttpRetryOptions = MagicMock(side_effect=lambda **kw: {"retry_options": kw})  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_response = MagicMock(output_text="Gemini generated response", id="int_123", steps=[])
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="key")
    request = GenerationRequest(
        messages=[
            Message(role="system", content="gemini system"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="prior reply"),
        ]
    )

    response = provider.generate(request)

    assert response == GenerationResponse(id="int_123", text="Gemini generated response", model="gemini-3.8-flash")
    mock_genai.Client.assert_called_once_with(api_key="key")
    mock_client.interactions.create.assert_called_once_with(
        model="gemini-3.8-flash",
        input=[
            {"type": "user_input", "content": [{"type": "text", "text": "hello"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "prior reply"}]},
        ],
        system_instruction="gemini system",
    )


def test_gemini_provider_generate_no_system_and_none_text(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_types.HttpOptions = MagicMock(side_effect=lambda **kw: {"http_options": kw})  # type: ignore[attr-defined]
    mock_types.HttpRetryOptions = MagicMock(side_effect=lambda **kw: {"retry_options": kw})  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MagicMock(output_text=None, text="", id="int_none", steps=[])
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    response = provider.generate(GenerationRequest(prompt="hello"))

    assert response == GenerationResponse(id="int_none", text="", model="gemini-3.8-flash")
    assert "system_instruction" not in mock_client.interactions.create.call_args.kwargs



def test_gemini_provider_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    provider = GeminiProvider(model="gemini-2.5-flash")
    with pytest.raises(ImportError, match="Install Gemini support with: pip install 'evidor\\[gemini\\]'"):
        provider.generate(GenerationRequest(prompt="test"))


# --- Provider Tool Calling Tests ---


def test_openai_provider_responses_api_with_tool_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_tc = MagicMock()
    mock_tc.type = "function_call"
    mock_tc.id = "call_abc"
    mock_tc.name = "get_weather"
    mock_tc.arguments = '{"location": "Tokyo"}'

    mock_response = MagicMock(output_text="Checking weather", output=[mock_tc], id="resp_tool_1")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", api_key="test-key")
    tool_def = Tool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        func=lambda location: "sunny",
    )
    request = GenerationRequest(
        messages=[
            Message(role="user", content="What's the weather in Tokyo?"),
            Message(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="call_abc", name="get_weather", arguments={"location": "Tokyo"}),),
            ),
            Message(role="tool", content="sunny", tool_call_id="call_abc"),
        ],
        tools=(tool_def,),
    )
    resp = provider.generate(request)
    assert resp.text == "Checking weather"
    assert resp.id == "resp_tool_1"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0] == ToolCall(id="call_abc", name="get_weather", arguments={"location": "Tokyo"})

    call_args = mock_client.responses.create.call_args.kwargs
    assert call_args["model"] == "gpt-6-astra"
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["name"] == "get_weather"
    assert call_args["input"][1] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "get_weather",
        "arguments": '{"location": "Tokyo"}',
    }
    assert call_args["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "sunny",
    }


def test_openai_provider_responses_api_with_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_output_item = MagicMock()
    mock_output_item.type = "function_call"
    mock_output_item.name = "calc"
    mock_output_item.arguments = '{"x": 5}'
    mock_output_item.call_id = "call_xyz"
    mock_response = MagicMock(output_text="Calling calc", output=[mock_output_item])
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-4.1-mini")
    tool_def = Tool(
        name="calc",
        description="calculate",
        parameters={"type": "object"},
        func=lambda x: x,
    )
    request = GenerationRequest(
        messages=[
            Message(role="user", content="compute"),
            Message(
                role="assistant",
                content="thinking",
                tool_calls=(ToolCall(id="call_xyz", name="calc", arguments={"x": 5}),),
            ),
            Message(role="tool", content="5", tool_call_id="call_xyz"),
        ],
        tools=(tool_def,),
    )
    resp = provider.generate(request)
    assert resp.text == "Calling calc"
    assert resp.tool_calls == (ToolCall(id="call_xyz", name="calc", arguments={"x": 5}),)


def test_anthropic_provider_tools_and_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_tool_use_block = MagicMock()
    mock_tool_use_block.type = "tool_use"
    mock_tool_use_block.id = "call_anthropic_1"
    mock_tool_use_block.name = "search"
    mock_tool_use_block.input = {"q": "python"}
    mock_text_block = MagicMock(type="text", text="I will search")
    mock_message = MagicMock(content=[mock_text_block, mock_tool_use_block])
    mock_client.messages.create.return_value = mock_message
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-5", api_key="key")
    tool_def = Tool(
        name="search",
        description="Search web",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        func=lambda q: "results",
    )
    request = GenerationRequest(
        messages=[
            Message(role="user", content="Search python"),
            Message(
                role="assistant",
                content="Searching...",
                tool_calls=(ToolCall(id="call_anthropic_1", name="search", arguments={"q": "python"}),),
            ),
            Message(role="tool", content="results", tool_call_id="call_anthropic_1"),
        ],
        tools=(tool_def,),
    )
    resp = provider.generate(request)
    assert resp.text == "I will search"
    assert resp.tool_calls == (ToolCall(id="call_anthropic_1", name="search", arguments={"q": "python"}),)

    create_kwargs = mock_client.messages.create.call_args.kwargs
    assert len(create_kwargs["tools"]) == 1
    assert create_kwargs["tools"][0]["name"] == "search"
    assert create_kwargs["tools"][0]["input_schema"] == tool_def.parameters
    assert create_kwargs["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_anthropic_1", "content": "results"}],
    }


def test_gemini_provider_tools_and_function_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_types.HttpOptions = MagicMock(side_effect=lambda **kw: {"http_options": kw})  # type: ignore[attr-defined]
    mock_types.HttpRetryOptions = MagicMock(side_effect=lambda **kw: {"retry_options": kw})  # type: ignore[attr-defined]

    mock_step = {
        "type": "function_call",
        "id": "call_gemini_1",
        "name": "lookup",
        "arguments": {"id": 42},
    }
    mock_response = MagicMock(output_text="Looking it up", steps=[mock_step], id="int_tool_1")
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    tool_def = Tool(
        name="lookup",
        description="Lookup ID",
        parameters={"type": "object"},
        func=lambda id: "data",
    )
    request = GenerationRequest(
        messages=[
            Message(role="user", content="lookup 42"),
            Message(
                role="assistant",
                content="ok",
                tool_calls=(ToolCall(id="call_gemini_1", name="lookup", arguments={"id": 42}),),
            ),
            Message(role="tool", content="data", name="lookup", tool_call_id="call_gemini_1"),
        ],
        tools=(tool_def,),
    )
    resp = provider.generate(request)
    assert resp.text == "Looking it up"
    assert resp.id == "int_tool_1"
    assert resp.tool_calls == (ToolCall(id="call_gemini_1", name="lookup", arguments={"id": 42}),)


def test_gemini_provider_sanitizes_vendor_mcp_extensions_in_tool_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_types.HttpOptions = MagicMock(side_effect=lambda **kw: {"http_options": kw})  # type: ignore[attr-defined]
    mock_types.HttpRetryOptions = MagicMock(side_effect=lambda **kw: {"retry_options": kw})  # type: ignore[attr-defined]

    mock_response = MagicMock(output_text="Triage done", steps=[], id="int_schema_1")
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    tool_def = Tool(
        name="github_mcp_tool",
        description="GitHub tool",
        parameters={
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "x-custom-meta": "value",
            "properties": {
                "owner": {
                    "type": "string",
                    "x-mcp-header": "owner",
                    "description": "Repo owner",
                },
                "repo": {
                    "type": "string",
                    "x-mcp-header": "repo",
                },
                "issue_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": ["string", "number", "boolean"],
                                "description": "The field value",
                            }
                        },
                    },
                },
            },
            "required": ["owner", "repo"],
        },
        func=lambda owner, repo, issue_fields=None: "ok",
    )
    request = GenerationRequest(
        messages=[Message(role="user", content="triage")],
        tools=(tool_def,),
    )
    resp = provider.generate(request)
    assert resp.text == "Triage done"

    call_tools = mock_client.interactions.create.call_args.kwargs["tools"]
    sanitized = call_tools[0]["parameters"]
    assert "$schema" not in sanitized
    assert "x-custom-meta" not in sanitized
    assert "x-mcp-header" not in sanitized["properties"]["owner"]
    assert "x-mcp-header" not in sanitized["properties"]["repo"]
    assert sanitized["properties"]["owner"]["description"] == "Repo owner"

    # Multi-type list ['string', 'number', 'boolean'] should be converted to anyOf
    value_schema = sanitized["properties"]["issue_fields"]["items"]["properties"]["value"]
    assert "type" not in value_schema
    assert value_schema["anyOf"] == [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]
    assert value_schema["description"] == "The field value"


def test_gemini_provider_function_call_via_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_fc_step = MagicMock(
        type="function_call",
        id="call_gemini_2",
        arguments={"term": "python"},
        signature=None,
    )
    mock_fc_step.name = "query"
    mock_response = MagicMock(output_text="", steps=[mock_fc_step], usage=None)

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    resp = provider.generate(GenerationRequest(prompt="query python"))
    assert resp.text == ""
    assert resp.tool_calls == (ToolCall(id="call_gemini_2", name="query", arguments={"term": "python"}),)


def test_gemini_provider_multiple_function_calls_avoids_response_text(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_steps = []
    for i in range(1, 5):
        fc = MagicMock(
            type="function_call",
            id=f"fc_{i}",
            arguments={"arg": i},
            signature=None,
        )
        fc.name = f"tool_{i}"
        mock_steps.append(fc)

    mock_response = MagicMock(output_text="", steps=mock_steps, usage=None)
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    resp = provider.generate(GenerationRequest(prompt="run multiple tools"))
    assert resp.text == ""
    assert len(resp.tool_calls) == 4
    for i in range(1, 5):
        assert resp.tool_calls[i - 1] == ToolCall(id=f"fc_{i}", name=f"tool_{i}", arguments={"arg": i})


def test_gemini_provider_text_and_tools_with_thoughts(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    thought_step = MagicMock(
        type="thought",
        signature="sig_1",
        summary=[MagicMock(text="Let me plan this step.")],
    )
    model_output_step = MagicMock(
        type="model_output",
        content=[MagicMock(text="Here is what I will do.")],
    )
    fc_step = MagicMock(
        type="function_call",
        id="call_99",
        arguments={"x": 10},
        signature=None,
    )
    fc_step.name = "calculate"

    mock_response = MagicMock(
        output_text="Here is what I will do.",
        steps=[thought_step, model_output_step, fc_step],
        usage=None,
    )
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash")
    resp = provider.generate(GenerationRequest(prompt="do something"))
    assert resp.text == "Here is what I will do."
    assert resp.tool_calls == (ToolCall(id="call_99", name="calculate", arguments={"x": 10}),)
    assert resp.thinking == "Let me plan this step."



def test_openai_provider_handles_malformed_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.agent.providers.base import safe_parse_tool_arguments

    assert safe_parse_tool_arguments(None) == {}
    assert safe_parse_tool_arguments({"already": "dict"}) == {"already": "dict"}
    assert safe_parse_tool_arguments('{"valid": 123}') == {"valid": 123}
    assert safe_parse_tool_arguments('"string"') == {"__raw_args__": "string"}
    
    malformed = safe_parse_tool_arguments("{bad json")
    assert "__decode_error__" in malformed
    assert malformed["__raw_args__"] == "{bad json"

    # Test through responses.create
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_tc = MagicMock()
    mock_tc.type = "function_call"
    mock_tc.id = "c_bad"
    mock_tc.name = "broken"
    mock_tc.arguments = "{unclosed json"

    mock_response = MagicMock(output_text="test", output=[mock_tc], id="resp_bad")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra")
    resp = provider.generate(GenerationRequest(prompt="call broken"))
    assert len(resp.tool_calls) == 1
    assert "__decode_error__" in resp.tool_calls[0].arguments
    assert resp.tool_calls[0].arguments["__raw_args__"] == "{unclosed json"


# --- Prompt Caching Tests ---


def test_anthropic_provider_prompt_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_text_block = MagicMock(type="text", text="cached response")
    mock_usage = MagicMock(
        input_tokens=50,
        output_tokens=25,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=200,
    )
    mock_response = MagicMock(content=[mock_text_block], usage=mock_usage)
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    tool_b = Tool(name="b_tool", description="Tool B", parameters={}, func=lambda: "b")
    tool_a = Tool(name="a_tool", description="Tool A", parameters={}, func=lambda: "a")

    provider = AnthropicProvider(model="claude-3-5-sonnet", enable_prompt_caching=True)
    assert provider.enable_prompt_caching is True

    copied = provider.with_model("claude-3-5-haiku")
    assert copied.enable_prompt_caching is True

    req = GenerationRequest(
        messages=[
            Message(role="system", content="System instructions"),
            Message(role="user", content="Turn 1 prompt"),
            Message(role="assistant", content="Turn 1 reply"),
            Message(role="user", content="Turn 2 prompt"),
        ],
        tools=[tool_b, tool_a],
    )

    resp = provider.generate(req)
    assert resp.text == "cached response"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 1050
    assert resp.usage.completion_tokens == 25
    assert resp.usage.cached_tokens == 1000
    assert resp.usage.cache_creation_tokens == 200

    create_kwargs = mock_client.messages.create.call_args.kwargs
    # Top-level automatic caching
    assert create_kwargs["cache_control"] == {"type": "ephemeral"}
    assert create_kwargs["system"] == "System instructions"
    assert len(create_kwargs["tools"]) == 2
    assert create_kwargs["tools"][0]["name"] == "a_tool"
    assert create_kwargs["tools"][1]["name"] == "b_tool"

    messages = create_kwargs["messages"]
    assert len(messages) == 3
    assert messages[0]["content"] == "Turn 1 prompt"
    assert messages[1]["content"] == "Turn 1 reply"
    assert messages[2]["content"] == "Turn 2 prompt"


def test_anthropic_provider_disable_prompt_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(content=[MagicMock(type="text", text="ok")], usage=None)
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-5", enable_prompt_caching=False)
    assert provider.enable_prompt_caching is False

    req = GenerationRequest(
        messages=[
            Message(role="user", content="hello"),
        ]
    )
    provider.generate(req)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "cache_control" not in call_kwargs


def test_openai_provider_prompt_caching_and_tool_sorting(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_usage = MagicMock(
        prompt_tokens=1200,
        completion_tokens=50,
        total_tokens=1250,
        prompt_tokens_details=MagicMock(cached_tokens=1024),
    )
    mock_response = MagicMock(output_text="OpenAI reply", output=[], usage=mock_usage, id="resp_cache")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    tool_z = Tool(name="z_tool", description="Z", parameters={}, func=lambda: "z")
    tool_a = Tool(name="a_tool", description="A", parameters={}, func=lambda: "a")

    provider = OpenAIProvider(model="gpt-6-astra")
    req = GenerationRequest(
        messages=[Message(role="user", content="hello")],
        tools=[tool_z, tool_a],
    )
    resp = provider.generate(req)

    assert resp.text == "OpenAI reply"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 1200
    assert resp.usage.cached_tokens == 1024

    create_kwargs = mock_client.responses.create.call_args.kwargs
    tools = create_kwargs["tools"]
    assert tools[0]["name"] == "a_tool"
    assert tools[1]["name"] == "z_tool"


def test_openai_provider_deepseek_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_usage = MagicMock(
        prompt_tokens=1500,
        completion_tokens=100,
        total_tokens=1600,
        prompt_tokens_details=MagicMock(cached_tokens=0, prompt_cache_hit_tokens=1400),
    )
    mock_response = MagicMock(output_text="DeepSeek reply", output=[], usage=mock_usage, id="resp_ds")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra")
    resp = provider.generate(GenerationRequest(prompt="hi"))

    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 1500
    assert resp.usage.cached_tokens == 1400


def test_gemini_provider_prompt_caching_and_cached_content(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_usage = MagicMock(
        total_input_tokens=2000,
        total_output_tokens=150,
        total_tokens=2150,
        total_cached_tokens=1800,
        total_thought_tokens=0,
    )
    mock_client = MagicMock()
    mock_response = MagicMock(
        output_text="Gemini cached reply",
        steps=[],
        usage=mock_usage,
    )
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", cached_content="cachedContents/abc123xyz")
    assert provider.cached_content == "cachedContents/abc123xyz"

    copied = provider.with_model("gemini-3.5-flash-lite")
    assert copied.cached_content == "cachedContents/abc123xyz"

    resp = provider.generate(GenerationRequest(prompt="hello cached gemini"))

    assert resp.text == "Gemini cached reply"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 2000
    assert resp.usage.cached_tokens == 1800

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["cached_content"] == "cachedContents/abc123xyz"


# --- Gemini Interactions API Tests ---


def test_gemini_provider_interactions_api_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()
    mock_usage = MagicMock(
        total_input_tokens=15,
        total_output_tokens=25,
        total_tokens=40,
        total_cached_tokens=5,
        total_thought_tokens=0,
    )
    mock_response = MagicMock(
        output_text="Gemini Interactions API response",
        steps=[],
        usage=mock_usage,
    )
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="test-key")
    request = GenerationRequest(
        messages=[
            Message(role="system", content="System instruction"),
            Message(role="user", content="Hello Gemini"),
        ]
    )

    response = provider.generate(request)

    assert response.text == "Gemini Interactions API response"
    assert response.model == "gemini-3.8-flash"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.completion_tokens == 25
    assert response.usage.total_tokens == 40
    assert response.usage.cached_tokens == 5

    mock_client.interactions.create.assert_called_once_with(
        model="gemini-3.8-flash",
        input=[{"type": "user_input", "content": [{"type": "text", "text": "Hello Gemini"}]}],
        system_instruction="System instruction",
    )


def test_gemini_provider_interactions_api_with_tools_and_multi_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_fc_step = MagicMock(
        type="function_call",
        id="call_gemini_int_1",
        arguments={"user_id": 123},
        signature="fc_sig_123",
    )
    mock_fc_step.name = "lookup_user"
    mock_response = MagicMock(
        output_text="",
        steps=[mock_fc_step],
        usage=None,
    )

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    tool_def = Tool(
        name="lookup_user",
        description="Lookup user by ID",
        parameters={"type": "object", "properties": {"user_id": {"type": "integer"}}},
        func=lambda user_id: "user data",
    )

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="test-key")
    request = GenerationRequest(
        messages=[
            Message(role="user", content="Find user 123"),
            Message(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="call_gemini_int_1", name="lookup_user", arguments={"user_id": 123}),),
            ),
            Message(role="tool", content="found Alice", name="lookup_user", tool_call_id="call_gemini_int_1"),
        ],
        tools=(tool_def,),
    )

    response = provider.generate(request)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0] == ToolCall(id="call_gemini_int_1", name="lookup_user", arguments={"user_id": 123})
    assert len(response.thought_blocks) == 1
    assert response.thought_blocks[0]["type"] == "function_call"

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["name"] == "lookup_user"
    assert call_kwargs["tools"][0]["type"] == "function"
    steps = call_kwargs["input"]
    assert steps[0] == {"type": "user_input", "content": [{"type": "text", "text": "Find user 123"}]}
    assert steps[1] == {"type": "function_call", "id": "call_gemini_int_1", "name": "lookup_user", "arguments": {"user_id": 123}}
    assert steps[2] == {"type": "function_result", "call_id": "call_gemini_int_1", "name": "lookup_user", "result": {"result": "found Alice"}}


def test_gemini_provider_interactions_api_with_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.models import GeminiThinkingConfig

    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_thought_content = MagicMock(text="Analyzing user query step by step.")
    mock_thought_step = MagicMock(
        type="thought",
        signature="thought_sig_999",
        summary=[mock_thought_content],
    )
    mock_output_content = MagicMock(text="Here is the detailed analysis.")
    mock_model_output_step = MagicMock(
        type="model_output",
        content=[mock_output_content],
    )
    mock_usage = MagicMock(
        total_input_tokens=100,
        total_output_tokens=200,
        total_tokens=300,
        total_cached_tokens=0,
        total_thought_tokens=80,
    )
    mock_response = MagicMock(
        output_text="Here is the detailed analysis.",
        steps=[mock_thought_step, mock_model_output_step],
        usage=mock_usage,
    )

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(
        model="gemini-3.8-flash",
        thinking=GeminiThinkingConfig(thinking_level="high", include_thoughts=True),
    )
    request = GenerationRequest(prompt="Analyze this")
    response = provider.generate(request)

    assert response.text == "Here is the detailed analysis."
    assert response.thinking == "Analyzing user query step by step."
    assert len(response.thought_blocks) == 1
    assert response.thought_blocks[0]["type"] == "thought"
    assert response.thought_blocks[0]["text"] == "Analyzing user query step by step."
    assert response.thought_blocks[0]["thought_signature"] == "thought_sig_999"
    assert response.usage is not None
    assert response.usage.reasoning_tokens == 80

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["generation_config"] == {"thinking_level": "high", "thinking_summaries": "auto"}


# --- Compaction Tests ---


def test_openai_provider_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_compaction_item = MagicMock(type="compaction", content="Encrypted compaction state")
    mock_response = MagicMock(
        id="resp_compact_1",
        output_text="Answer after compaction",
        output=[mock_compaction_item],
        usage=None,
    )
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-6-astra", api_key="sk-test", compact_threshold=100000)
    assert provider.compact_threshold == 100000

    new_provider = provider.with_model("gpt-6.1-sol")
    assert new_provider.compact_threshold == 100000

    request = GenerationRequest(prompt="Long context turn")
    response = provider.generate(request)

    assert response.text == "Answer after compaction"
    assert response.compaction == "Encrypted compaction state"

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 100000}
    ]


def test_anthropic_provider_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_compaction_block = MagicMock(
        type="compaction",
        content="Summary: user requested web scraper.",
    )
    mock_text_block = MagicMock(
        type="text",
        text="Here is the continuation.",
    )
    mock_response = MagicMock(
        id="msg_compact_1",
        content=[mock_compaction_block, mock_text_block],
        usage=None,
    )
    mock_client.beta.messages.create.return_value = mock_response
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(
        model="claude-3-7-sonnet-20250219",
        api_key="ant-key",
        compact_threshold=150000,
        compact_instructions="Focus on architecture decisions.",
    )
    assert provider.compact_threshold == 150000
    assert provider.compact_instructions == "Focus on architecture decisions."

    new_provider = provider.with_model("claude-opus-4-6")
    assert new_provider.compact_threshold == 150000
    assert new_provider.compact_instructions == "Focus on architecture decisions."

    request = GenerationRequest(prompt="Begin complex refactor")
    response = provider.generate(request)

    assert response.text == "Here is the continuation."
    assert response.compaction == "Summary: user requested web scraper."

    call_kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert call_kwargs["betas"] == ["compact-2026-01-12"]
    assert call_kwargs["context_management"] == {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 150000},
                "instructions": "Focus on architecture decisions.",
            }
        ]
    }

    # Verify that formatting replayed messages puts compaction block first
    replayed_messages = [
        Message(role="user", content="Begin complex refactor"),
        Message(
            role="assistant",
            content="Here is the continuation.",
            compaction=response.compaction,
        ),
        Message(role="user", content="Next step"),
    ]
    formatted = provider._format_messages(replayed_messages)
    assert len(formatted) == 3
    assistant_formatted = formatted[1]
    assert assistant_formatted["role"] == "assistant"
    assert isinstance(assistant_formatted["content"], list)
    assert assistant_formatted["content"][0] == {
        "type": "compaction",
        "content": "Summary: user requested web scraper.",
    }
    assert assistant_formatted["content"][1] == {
        "type": "text",
        "text": "Here is the continuation.",
    }


def test_gemini_client_reuse_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")
    mock_client = MagicMock()
    mock_response = MagicMock(interaction=MagicMock(id="int_1", model="gemini-3.8-flash", steps=[]))
    mock_client.interactions.create.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="gem_key")
    req = GenerationRequest(prompt="hello")

    provider.generate(req)
    provider.generate(req)

    # Client is initialized only once across multiple generation calls
    assert mock_genai.Client.call_count == 1
    assert mock_client.interactions.create.call_count == 2

    # with_model preserves the cached client
    new_provider = provider.with_model("gemini-3.5-flash-lite")
    assert new_provider._client is mock_client

    provider.close()
    assert provider._client is None


def test_openai_client_reuse_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_response = MagicMock(output_text="Text")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-4.1-mini", api_key="key")
    req = GenerationRequest(prompt="hello")

    provider.generate(req)
    provider.generate(req)

    # Sync client is reused
    assert mock_openai.OpenAI.call_count == 1
    assert mock_client.responses.create.call_count == 2

    new_provider = provider.with_model("gpt-6-astra")
    assert new_provider._sync_client is mock_client

    provider.close()
    assert provider._sync_client is None


def test_anthropic_client_reuse_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_response = MagicMock(content=[MagicMock(type="text", text="Hello")], model="claude-3-5")
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-5", api_key="key")
    req = GenerationRequest(prompt="hello")

    provider.generate(req)
    provider.generate(req)

    # Sync client is reused
    assert mock_anthropic.Anthropic.call_count == 1
    assert mock_client.messages.create.call_count == 2

    new_provider = provider.with_model("claude-3-7-sonnet")
    assert new_provider._sync_client is mock_client

    provider.close()
    assert provider._sync_client is None




