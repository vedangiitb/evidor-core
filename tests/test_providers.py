import asyncio
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
    provider = OpenAIProvider(model="gpt-4.1", api_key="sk-test", base_url="https://api.deepseek.com")
    assert provider.model == "gpt-4.1"
    assert provider.base_url == "https://api.deepseek.com"

    new_provider = provider.with_model("gpt-4.1-mini")
    assert new_provider.model == "gpt-4.1-mini"
    assert new_provider._api_key == "sk-test"
    assert new_provider.base_url == "https://api.deepseek.com"


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


def test_openai_provider_generate_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_response = MagicMock(output_text="DeepSeek text")
    mock_client.responses.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(
        model="deepseek-flash",
        api_key="ds-key",
        base_url="https://api.deepseek.com",
    )
    request = GenerationRequest(prompt="Hello DeepSeek")
    response = provider.generate(request)

    assert response.text == "DeepSeek text"
    mock_openai.OpenAI.assert_called_once_with(api_key="ds-key", base_url="https://api.deepseek.com")


@pytest.mark.asyncio
async def test_openai_provider_async_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_response = MagicMock(output_text="DeepSeek async text")

    async def _async_create(**kwargs: Any) -> Any:
        return mock_response

    mock_client.responses.create = _async_create
    mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(
        model="deepseek-flash",
        api_key="ds-key",
        base_url="https://api.deepseek.com",
    )
    request = GenerationRequest(prompt="Hello DeepSeek async")
    response = await provider.generate_async(request)

    assert response.text == "DeepSeek async text"
    mock_openai.AsyncOpenAI.assert_called_once_with(api_key="ds-key", base_url="https://api.deepseek.com")




def test_openai_provider_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = OpenAIProvider(model="gpt-4.1-mini")
    with pytest.raises(ImportError, match="Install OpenAI support with: pip install 'evidor\\[openai\\]'"):
        provider.generate(GenerationRequest(prompt="test"))


@pytest.mark.asyncio
async def test_openai_provider_async_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = OpenAIProvider(model="gpt-4.1-mini")
    with pytest.raises(ImportError, match="Install OpenAI support with: pip install 'evidor\\[openai\\]'"):
        await provider.generate_async(GenerationRequest(prompt="test"))


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
    assert "cached_content" not in call_kwargs


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
        thinking=GeminiThinkingConfig(thinking_level="high", thinking_summaries="auto"),
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

    # with_model decouples client ownership so closing one does not invalidate the other
    new_provider = provider.with_model("gemini-3.5-flash-lite")
    provider.close()
    assert provider._client is None

    # Clone can generate independently
    new_provider.generate(req)
    assert mock_genai.Client.call_count == 2
    new_provider.close()
    assert new_provider._client is None


@pytest.mark.asyncio
async def test_gemini_client_close_async_and_reinstantiation(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_client = MagicMock()
    mock_aio = MagicMock()
    mock_aio.aclose = AsyncMock()
    mock_response = MagicMock(interaction=MagicMock(id="int_1", model="gemini-3.8-flash", steps=[]))
    mock_aio.interactions = MagicMock()
    mock_aio.interactions.create = AsyncMock(return_value=mock_response)
    mock_client.aio = mock_aio
    mock_client.interactions.create.return_value = mock_response

    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-3.8-flash", api_key="gem_key")
    req = GenerationRequest(prompt="hello")

    await provider.generate_async(req)
    assert mock_genai.Client.call_count == 1

    # 4.1: close_async calls aclose() on aio_client
    await provider.close_async()
    mock_aio.aclose.assert_awaited_once()
    assert provider._client is None

    # Reconnect and generate again
    await provider.generate_async(req)
    assert mock_genai.Client.call_count == 2

    # 4.2: Missing re-instantiation check on closed client
    # Simulate client connection being closed externally (e.g. is_closed property)
    provider._client.is_closed = True
    await provider.generate_async(req)
    assert mock_genai.Client.call_count == 3



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

    # with_model decouples client ownership so closing one does not invalidate the other
    new_provider = provider.with_model("gpt-6-astra")
    provider.close()
    assert provider._sync_client is None

    # Clone can generate independently
    new_provider.generate(req)
    assert mock_openai.OpenAI.call_count == 2
    new_provider.close()
    assert new_provider._sync_client is None


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

    # with_model decouples client ownership so closing one does not invalidate the other
    new_provider = provider.with_model("claude-3-7-sonnet")
    provider.close()
    assert provider._sync_client is None

    # Clone can generate independently
    new_provider.generate(req)
    assert mock_anthropic.Anthropic.call_count == 2
    new_provider.close()
    assert new_provider._sync_client is None


def test_openai_prunes_closed_event_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-4.1-mini", api_key="key")

    async def _fetch():
        return provider._get_async_client()

    loop1 = asyncio.new_event_loop()
    client1 = loop1.run_until_complete(_fetch())
    assert loop1 in provider._async_clients
    loop1.close()

    # Now on loop2, getting a client should prune loop1
    loop2 = asyncio.new_event_loop()
    client2 = loop2.run_until_complete(_fetch())
    assert loop1 not in provider._async_clients
    assert loop2 in provider._async_clients
    loop2.close()


def test_is_client_closed_detection() -> None:
    from evidor.agent.providers.base import is_client_closed

    class BoolPropClient:
        is_closed = True

    class OpenBoolPropClient:
        is_closed = False

    class MethodClient:
        def __init__(self, closed: bool) -> None:
            self._closed = closed
        def is_closed(self) -> bool:
            return self._closed

    assert is_client_closed(BoolPropClient()) is True
    assert is_client_closed(OpenBoolPropClient()) is False
    assert is_client_closed(MethodClient(True)) is True
    assert is_client_closed(MethodClient(False)) is False
    assert is_client_closed(MagicMock()) is False


def test_gemini_payload_builder_step_ordering() -> None:
    from evidor.agent.providers.gemini.builder import GeminiPayloadBuilder
    from evidor.models import Message, ToolCall

    builder = GeminiPayloadBuilder(model="gemini-3.8-flash")
    msg = Message(
        role="assistant",
        content="I will call the search tool now.",
        thought_blocks=[
            {
                "type": "thought",
                "text": "User wants current weather data.",
                "thought_signature": "sig_thought_1",
            },
            {
                "type": "function_call",
                "id": "call_1",
                "name": "search",
                "args": {"q": "weather in NYC"},
                "thought_signature": "sig_fc_1",
            },
        ],
        tool_calls=(
            ToolCall(id="call_1", name="search", arguments={"q": "weather in NYC"}),
            ToolCall(id="call_2", name="lookup", arguments={"city": "NYC"}),
        ),
    )

    steps = builder.format_input_steps([msg])
    assert len(steps) == 4
    # Step 1: thought
    assert steps[0]["type"] == "thought"
    assert steps[0]["signature"] == "sig_thought_1"
    assert steps[0]["summary"] == [{"type": "text", "text": "User wants current weather data."}]
    # Step 2: model_output (text content) comes BEFORE function calls
    assert steps[1]["type"] == "model_output"
    assert steps[1]["content"] == [{"type": "text", "text": "I will call the search tool now."}]
    # Step 3: function_call with signature from thought_blocks
    assert steps[2]["type"] == "function_call"
    assert steps[2]["id"] == "call_1"
    assert steps[2]["signature"] == "sig_fc_1"
    # Step 4: additional function_call from tool_calls
    assert steps[3]["type"] == "function_call"
    assert steps[3]["id"] == "call_2"


def test_gemini_payload_builder_thinking_config() -> None:
    from evidor.agent.providers.gemini.builder import GeminiPayloadBuilder
    from evidor.models import GeminiThinkingConfig, GenerationRequest

    req = GenerationRequest(prompt="hello")

    # 1. Arbitrary / future thinking_level (no model check, no whitelist)
    b_custom = GeminiPayloadBuilder(
        model="gemini-3.8-flash",
        thinking=GeminiThinkingConfig(thinking_level="experimental_deep", thinking_summaries="auto"),
    )
    p_custom = b_custom.build(req)
    assert p_custom["generation_config"] == {
        "thinking_level": "experimental_deep",
        "thinking_summaries": "auto",
    }

    # 2. thinking_summaries="none"
    b_no_thoughts = GeminiPayloadBuilder(
        model="gemini-3.8-flash",
        thinking=GeminiThinkingConfig(thinking_level="high", thinking_summaries="none"),
    )
    p_no_thoughts = b_no_thoughts.build(req)
    assert p_no_thoughts["generation_config"] == {
        "thinking_level": "high",
        "thinking_summaries": "none",
    }

    # 3. enabled=False disabled thinking -> minimal + none
    b_off = GeminiPayloadBuilder(
        model="gemini-3.8-flash",
        thinking=GeminiThinkingConfig(enabled=False),
    )
    p_off = b_off.build(req)
    assert p_off["generation_config"] == {
        "thinking_level": "minimal",
        "thinking_summaries": "none",
    }


def test_gemini_payload_builder_types_optional() -> None:
    from evidor.agent.providers.gemini.builder import GeminiPayloadBuilder
    from evidor.models import GenerationRequest
    from evidor.tools import Tool

    builder = GeminiPayloadBuilder(model="gemini-3.8-flash")

    # _sanitize_schema works as an internal instance method
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    sanitized = builder._sanitize_schema(schema)
    assert sanitized["type"] == "object"
    assert "name" in sanitized["properties"]

    # builder.build works without types argument
    tool = Tool(name="test_tool", description="A test tool", parameters=schema, func=lambda: None)
    req = GenerationRequest(prompt="Use tool", tools=[tool])
    payload = builder.build(req)
    assert "cached_content" not in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "test_tool"


def test_gemini_schema_sanitizer_enhancements() -> None:
    from evidor.agent.providers.gemini.builder import GeminiPayloadBuilder

    builder = GeminiPayloadBuilder(model="gemini-3.8-flash")

    # 1. Bare array missing items gets default string items
    schema_array = {
        "type": "object",
        "properties": {
            "tags": {"type": "array"},
        },
    }
    sanitized_array = builder._sanitize_schema(schema_array)
    assert sanitized_array["properties"]["tags"]["items"] == {"type": "string"}

    # 2. Empty required list is omitted
    schema_empty_req = {
        "type": "object",
        "properties": {"opt": {"type": "string"}},
        "required": [],
    }
    sanitized_empty_req = builder._sanitize_schema(schema_empty_req)
    assert "required" not in sanitized_empty_req

    # 3. Non-empty required list is preserved
    schema_with_req = {
        "type": "object",
        "properties": {"req_field": {"type": "string"}},
        "required": ["req_field"],
    }
    sanitized_with_req = builder._sanitize_schema(schema_with_req)
    assert sanitized_with_req["required"] == ["req_field"]

    # 4. Normalization of snake_case to camelCase
    schema_snake = {
        "type": "object",
        "properties": {
            "val": {
                "max_length": 50,
                "min_length": 5,
                "any_of": [{"type": "string"}, {"type": "number"}],
            }
        },
        "property_ordering": ["val"],
    }
    sanitized_snake = builder._sanitize_schema(schema_snake)
    val_prop = sanitized_snake["properties"]["val"]
    assert "maxLength" in val_prop
    assert "max_length" not in val_prop
    assert "minLength" in val_prop
    assert "min_length" not in val_prop
    assert "anyOf" in val_prop
    assert "any_of" not in val_prop
    assert "propertyOrdering" in sanitized_snake
    assert "property_ordering" not in sanitized_snake


def test_gemini_parser_no_thought_leakage() -> None:
    from evidor.agent.providers.gemini.parser import GeminiResponseParser

    parser = GeminiResponseParser(model="gemini-3.8-flash")
    raw_response = {
        "id": "int_thought_only",
        "output_text": "",
        "steps": [
            {
                "type": "thought",
                "summary": [{"type": "text", "text": "Deep thinking reasoning..."}],
                "signature": "sig_thought",
            }
        ],
    }
    resp = parser.parse(raw_response)
    # Thought should NOT leak into text
    assert resp.text == ""
    assert resp.thinking == "Deep thinking reasoning..."
    assert len(resp.thought_blocks) == 1
    assert resp.thought_blocks[0]["text"] == "Deep thinking reasoning..."


def test_gemini_parser_multi_step_model_output() -> None:
    from evidor.agent.providers.gemini.parser import GeminiResponseParser

    parser = GeminiResponseParser(model="gemini-3.8-flash")
    raw_response = {
        "id": "int_multi_step",
        "steps": [
            {
                "type": "thought",
                "summary": "Thinking first.",
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Part 1 of the answer. "}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Part 2 of the answer."}],
            },
        ],
    }
    resp = parser.parse(raw_response)
    assert resp.text == "Part 1 of the answer. Part 2 of the answer."
    assert resp.thinking == "Thinking first."


def test_gemini_parser_usage_dictionary_payload() -> None:
    from evidor.agent.providers.gemini.parser import GeminiResponseParser

    parser = GeminiResponseParser(model="gemini-3.8-flash")
    usage_dict = {
        "total_input_tokens": 120,
        "total_output_tokens": 80,
        "total_tokens": 200,
        "total_cached_tokens": 30,
        "total_thought_tokens": 40,
    }
    usage = parser.parse_usage(usage_dict)
    assert usage is not None
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 80
    assert usage.total_tokens == 200
    assert usage.cached_tokens == 30
    assert usage.reasoning_tokens == 40


def test_gemini_parser_thought_summary_dict() -> None:
    from evidor.agent.providers.gemini.parser import GeminiResponseParser

    parser = GeminiResponseParser(model="gemini-3.8-flash")
    raw_response = {
        "id": "int_dict_summary",
        "output_text": "Answer text",
        "steps": [
            {
                "type": "thought",
                "summary": {"type": "text", "text": "Dict summary text"},
                "signature": "sig_dict",
            }
        ],
    }
    resp = parser.parse(raw_response)
    assert resp.text == "Answer text"
    assert resp.thinking == "Dict summary text"
    assert len(resp.thought_blocks) == 1
    assert resp.thought_blocks[0]["thought_signature"] == "sig_dict"


def test_openai_payload_builder_compaction_and_thought_blocks() -> None:
    from evidor.agent.providers.openai.builder import OpenAIPayloadBuilder
    from evidor.models import GenerationRequest, Message

    builder = OpenAIPayloadBuilder(model="gpt-6-astra")
    req = GenerationRequest(
        messages=[
            Message(role="assistant", content="Prior summary", compaction="encrypted_compaction_state"),
            Message(
                role="assistant",
                content="Reasoned text",
                thought_blocks=({"type": "reasoning", "id": "rs_1", "encrypted_content": "enc_rs"},),
            ),
        ]
    )
    payload = builder.build(req)
    input_items = payload["input"]
    assert input_items[0] == {
        "type": "compaction",
        "encrypted_content": "encrypted_compaction_state",
    }
    assert input_items[1] == {"role": "assistant", "content": "Prior summary"}
    assert input_items[2] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc_rs"}
    assert input_items[3] == {"role": "assistant", "content": "Reasoned text"}


def test_openai_payload_builder_thinking_disabled_effort_none() -> None:
    from evidor.agent.providers.openai.builder import OpenAIPayloadBuilder
    from evidor.models import GenerationRequest, OpenAIThinkingConfig

    builder = OpenAIPayloadBuilder(model="gpt-6-astra", thinking=OpenAIThinkingConfig(enabled=False))
    req = GenerationRequest(prompt="No thinking please")
    payload = builder.build(req)
    assert payload["reasoning"] == {"effort": "none"}


def test_openai_payload_builder_assistant_content_precedes_tool_calls() -> None:
    from evidor.agent.providers.openai.builder import OpenAIPayloadBuilder
    from evidor.models import GenerationRequest, Message, ToolCall

    builder = OpenAIPayloadBuilder(model="gpt-6-astra")
    tc = ToolCall(id="call_999", name="lookup", arguments={"target": "val"})
    req = GenerationRequest(
        messages=[
            Message(role="assistant", content="Let me check that.", tool_calls=(tc,)),
            Message(role="tool", content="lookup_result", tool_call_id="call_999"),
        ]
    )
    payload = builder.build(req)
    input_items = payload["input"]
    # Assistant text must come before the function call so it doesn't get inserted between call and output
    assert input_items[0] == {"role": "assistant", "content": "Let me check that."}
    assert input_items[1] == {
        "type": "function_call",
        "call_id": "call_999",
        "name": "lookup",
        "arguments": '{"target": "val"}',
    }
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": "call_999",
        "output": "lookup_result",
    }


def test_openai_payload_builder_mutually_exclusive_chaining_ids() -> None:
    from evidor.agent.providers.openai.builder import OpenAIPayloadBuilder
    from evidor.models import GenerationRequest

    builder = OpenAIPayloadBuilder(model="gpt-6-astra")
    # Both provided -> previous_response_id takes precedence, conversation omitted per OpenAPI spec
    req = GenerationRequest(
        prompt="hello",
        previous_response_id="resp_123",
        conversation_id="conv_456",
    )
    payload = builder.build(req)
    assert "previous_response_id" in payload
    assert payload["previous_response_id"] == "resp_123"
    assert "conversation" not in payload


def test_openai_payload_builder_tool_arg_and_output_serialization() -> None:
    from evidor.agent.providers.openai.builder import OpenAIPayloadBuilder
    from evidor.models import GenerationRequest, Message, ToolCall

    builder = OpenAIPayloadBuilder(model="gpt-6-astra")
    tc_none = ToolCall(id="c_none", name="tool_a", arguments=None)  # type: ignore[arg-type]
    req = GenerationRequest(
        messages=[
            Message(role="assistant", tool_calls=(tc_none,)),
            Message(role="tool", content="ok", tool_call_id=None),
        ]
    )
    payload = builder.build(req)
    input_items = payload["input"]
    assert input_items[0]["arguments"] == "{}"
    assert input_items[1]["output"] == "ok"
    assert input_items[1]["call_id"] == "call_unknown"


# --- OpenAIResponseParser Tests (2.x fixes) ---


def test_openai_parser_dict_payload_and_missing_output_text() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser

    parser = OpenAIResponseParser(model="gpt-6-astra")
    # Wire JSON response from /v1/responses (no SDK output_text property)
    raw_response = {
        "id": "resp_json_001",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_001",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "output_text", "text": "world!"},
                ],
            },
            {
                "type": "function_call",
                "id": "fc_001",
                "call_id": "call_123",
                "name": "search_data",
                "arguments": '{"query": "evidor"}',
            },
        ],
        "usage": {
            "input_tokens": 15,
            "output_tokens": 25,
            "total_tokens": 40,
        },
    }

    res = parser.parse(raw_response)
    assert res.id == "resp_json_001"
    assert res.text == "Hello world!"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].id == "call_123"
    assert res.tool_calls[0].name == "search_data"
    assert res.tool_calls[0].arguments == {"query": "evidor"}
    assert res.usage is not None
    assert res.usage.prompt_tokens == 15
    assert res.usage.completion_tokens == 25
    assert res.usage.total_tokens == 40


def test_openai_parser_reasoning_items_and_thought_blocks() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser

    parser = OpenAIResponseParser(model="gpt-6-astra")
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_001",
        "summary": [{"type": "summary_text", "text": "Step 1: Analyzed input"}],
        "content": [{"type": "reasoning_text", "text": "Step 2: Computed intermediate result"}],
        "encrypted_content": "enc_reasoning_blob_789",
    }
    raw_response = {
        "id": "resp_reasoning_001",
        "output": [
            reasoning_item,
            {
                "type": "message",
                "id": "msg_002",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Final answer"}],
            },
        ],
    }

    res = parser.parse(raw_response)
    assert res.text == "Final answer"
    # Both summary and content must be preserved without dropping either
    assert res.thinking is not None
    assert "Step 1: Analyzed input" in res.thinking
    assert "Step 2: Computed intermediate result" in res.thinking
    # Preserves reasoning item in thought_blocks for stateless multi-turn replay
    assert len(res.thought_blocks) == 1
    assert res.thought_blocks[0] == reasoning_item


def test_openai_parser_top_level_reasoning_summary() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser

    parser = OpenAIResponseParser(model="gpt-6-astra")
    raw_response = {
        "id": "resp_top_reasoning",
        "output_text": "Answer",
        "reasoning": {
            "effort": "high",
            "summary": [{"type": "summary_text", "text": "Deliberating carefully..."}],
        },
    }

    res = parser.parse(raw_response)
    assert res.text == "Answer"
    assert res.thinking == "Deliberating carefully..."


def test_openai_parser_refusal_extraction() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser

    parser = OpenAIResponseParser(model="gpt-6-astra")
    raw_response = {
        "id": "resp_refusal_001",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "I cannot perform this unsafe action."},
                ],
            }
        ],
    }

    res = parser.parse(raw_response)
    assert res.text == "I cannot perform this unsafe action."


def test_openai_parser_usage_with_cache_write_tokens() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser

    parser = OpenAIResponseParser(model="gpt-6-astra")
    raw_response = {
        "id": "resp_usage_001",
        "output_text": "Usage reply",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_write_tokens": 70,
            },
            "output_tokens_details": {
                "reasoning_tokens": 20,
            },
        },
    }

    res = parser.parse(raw_response)
    assert res.usage is not None
    assert res.usage.prompt_tokens == 100
    assert res.usage.completion_tokens == 50
    assert res.usage.total_tokens == 150
    assert res.usage.cached_tokens == 30
    assert res.usage.cache_creation_tokens == 70
    assert res.usage.reasoning_tokens == 20


def test_openai_parser_compaction_extraction() -> None:
    from evidor.agent.providers.openai.parser import OpenAIResponseParser

    parser = OpenAIResponseParser(model="gpt-6-astra")
    raw_response = {
        "id": "resp_compact_001",
        "output": [
            {
                "type": "compaction",
                "encrypted_content": "enc_compact_token_xyz",
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Post compaction text"}],
            },
        ],
    }

    res = parser.parse(raw_response)
    assert res.text == "Post compaction text"
    assert res.compaction == "enc_compact_token_xyz"


# --- AnthropicPayloadBuilder Tests (1.x fixes) ---


def test_anthropic_builder_parallel_tool_results_merged_into_single_user_message() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet-20250219", max_tokens=2048)
    req = GenerationRequest(
        messages=[
            Message(role="user", content="Perform tasks"),
            Message(
                role="assistant",
                content="Running tools...",
                tool_calls=(
                    ToolCall(id="call_1", name="tool_a", arguments={"a": 1}),
                    ToolCall(id="call_2", name="tool_b", arguments={"b": 2}),
                ),
            ),
            Message(role="tool", content="result 1", tool_call_id="call_1"),
            Message(role="tool", content="result 2", tool_call_id="call_2"),
        ]
    )
    payload = builder.build(req)
    messages = payload["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert len(messages[2]["content"]) == 2
    assert messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "result 1",
    }
    assert messages[2]["content"][1] == {
        "type": "tool_result",
        "tool_use_id": "call_2",
        "content": "result 2",
    }


def test_anthropic_builder_tool_call_id_fallback_and_error() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet-20250219", max_tokens=2048)
    msg_err = Message(role="tool", content="Error occurred", tool_call_id=None, is_error=True)
    req = GenerationRequest(
        messages=[
            Message(role="user", content="Execute"),
            Message(
                role="assistant",
                content="Executing...",
                tool_calls=(ToolCall(id="call_x", name="fail_tool", arguments={}),),
            ),
            msg_err,
        ]
    )
    payload = builder.build(req)
    tool_content = payload["messages"][2]["content"]
    assert len(tool_content) == 1
    assert tool_content[0]["tool_use_id"] == "call_unknown"
    assert tool_content[0]["is_error"] is True
    assert tool_content[0]["content"] == "Error occurred"


def test_anthropic_builder_thought_blocks_sdk_objects_and_ordering() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet-20250219", max_tokens=2048)

    class MockSDKThinkingBlock:
        def __init__(self, thinking: str, signature: str) -> None:
            self.type = "thinking"
            self.thinking = thinking
            self.signature = signature

        def model_dump(self) -> dict[str, Any]:
            return {"type": "thinking", "thinking": self.thinking, "signature": self.signature}

    sdk_tb = MockSDKThinkingBlock("SDK reasoning", "sig_sdk_123")
    tc = ToolCall(id="call_calc", name="calc", arguments={"val": 10})
    req = GenerationRequest(
        messages=[
            Message(role="user", content="Calculate"),
            Message(
                role="assistant",
                content="Final text",
                tool_calls=(tc,),
                thought_blocks=(sdk_tb,),
            ),
        ]
    )
    payload = builder.build(req)
    assistant_blocks = payload["messages"][1]["content"]
    assert len(assistant_blocks) == 3
    # 1st block: thinking block (must be first)
    assert assistant_blocks[0] == {
        "type": "thinking",
        "thinking": "SDK reasoning",
        "signature": "sig_sdk_123",
    }
    # 2nd block: text
    assert assistant_blocks[1] == {"type": "text", "text": "Final text"}
    # 3rd block: tool_use
    assert assistant_blocks[2] == {
        "type": "tool_use",
        "id": "call_calc",
        "name": "calc",
        "input": {"val": 10},
    }


def test_anthropic_builder_thinking_budget_and_max_tokens_clamping() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder
    from evidor.models import AnthropicThinkingConfig

    # Budget tokens < 1024 auto-promoted to 1024, max_tokens bumped if <= budget
    cfg = AnthropicThinkingConfig(enabled=True, type="enabled", budget_tokens=500)
    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet-20250219", max_tokens=1000, thinking=cfg)
    req = GenerationRequest(prompt="Hello")
    payload = builder.build(req)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert payload["max_tokens"] > 1024
    assert payload["max_tokens"] == 1024 + 1024


def test_anthropic_builder_tool_schema_normalization() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet-20250219", max_tokens=2048)

    empty_param_tool = Tool(name="empty_param", description="Empty parameters", parameters=None, func=lambda: None)  # type: ignore[arg-type]
    dict_tool = {
        "name": "dict_tool",
        "description": "Defined as dictionary",
        "input_schema": {"type": "object", "properties": {"target": {"type": "string"}}},
    }

    req = GenerationRequest(
        prompt="Use tools",
        tools=[empty_param_tool, dict_tool],  # type: ignore[list-item]
    )
    payload = builder.build(req)
    tools = payload["tools"]
    assert len(tools) == 2
    # Dict tool sorted first alphabetically
    assert tools[0]["name"] == "dict_tool"
    assert tools[0]["description"] == "Defined as dictionary"
    assert tools[0]["input_schema"] == {"type": "object", "properties": {"target": {"type": "string"}}}

    # Empty param tool sorted second, normalized schema
    assert tools[1]["name"] == "empty_param"
    assert tools[1]["description"] == "Empty parameters"
    assert tools[1]["input_schema"] == {"type": "object", "properties": {}}


def test_anthropic_builder_consecutive_user_messages_merged() -> None:
    from evidor.agent.providers.anthropic.builder import AnthropicPayloadBuilder

    builder = AnthropicPayloadBuilder(model="claude-3-7-sonnet-20250219", max_tokens=2048)
    req = GenerationRequest(
        messages=[
            Message(role="user", content="Part 1"),
            Message(role="user", content="Part 2"),
        ]
    )
    payload = builder.build(req)
    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Part 1\n\nPart 2"


# --- AnthropicResponseParser Tests (2.x fixes) ---


def test_anthropic_parser_dict_payload() -> None:
    from evidor.agent.providers.anthropic.parser import AnthropicResponseParser

    parser = AnthropicResponseParser(model="claude-3-7-sonnet")
    dict_response = {
        "id": "msg_wire_001",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-20250219",
        "content": [
            {"type": "text", "text": "Wire text response"},
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        },
    }

    res = parser.parse(dict_response)
    assert res.id == "msg_wire_001"
    assert res.model == "claude-3-7-sonnet-20250219"
    assert res.text == "Wire text response"
    assert res.usage is not None
    assert res.usage.prompt_tokens == 140
    assert res.usage.completion_tokens == 45
    assert res.usage.cached_tokens == 20
    assert res.usage.cache_creation_tokens == 10


def test_anthropic_parser_tool_call_json_string_arguments_and_fallbacks() -> None:
    from evidor.agent.providers.anthropic.parser import AnthropicResponseParser

    parser = AnthropicResponseParser(model="claude-3-7-sonnet")
    raw_response = {
        "id": "msg_tool_002",
        "content": [
            {
                "type": "tool_use",
                "id": "call_weather_1",
                "name": "get_weather",
                "input": '{"location": "Tokyo", "days": 3}',
            },
            {
                "type": "tool_use",
                "id": "",  # Empty id fallback
                "name": "",  # Empty name fallback
                "input": None,
            },
        ],
    }

    res = parser.parse(raw_response)
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0].id == "call_weather_1"
    assert res.tool_calls[0].name == "get_weather"
    assert res.tool_calls[0].arguments == {"location": "Tokyo", "days": 3}

    assert res.tool_calls[1].id == "call_2"
    assert res.tool_calls[1].name == "unknown_tool"
    assert res.tool_calls[1].arguments == {}


def test_anthropic_parser_usage_with_thinking_tokens() -> None:
    from evidor.agent.providers.anthropic.parser import AnthropicResponseParser

    parser = AnthropicResponseParser(model="claude-3-7-sonnet")

    # Direct thinking_tokens in usage
    usage_with_thinking = {
        "input_tokens": 50,
        "output_tokens": 200,
        "thinking_tokens": 150,
    }
    res = parser.parse({"content": [], "usage": usage_with_thinking})
    assert res.usage is not None
    assert res.usage.prompt_tokens == 50
    assert res.usage.completion_tokens == 200
    assert res.usage.reasoning_tokens == 150

    # reasoning_tokens nested in output_tokens_details
    usage_with_nested = {
        "input_tokens": 60,
        "output_tokens": 180,
        "output_tokens_details": {"reasoning_tokens": 120},
    }
    res2 = parser.parse({"content": [], "usage": usage_with_nested})
    assert res2.usage is not None
    assert res2.usage.reasoning_tokens == 120


def test_anthropic_parser_redacted_thinking_and_string_content() -> None:
    from evidor.agent.providers.anthropic.parser import AnthropicResponseParser

    parser = AnthropicResponseParser(model="claude-3-7-sonnet")

    raw_response = {
        "id": "msg_redacted_003",
        "content": [
            {"type": "redacted_thinking", "data": "enc_data_blob_xyz"},
            {"type": "thinking", "thinking": "Internal thought", "signature": "sig_thought_1"},
            {"type": "text", "text": "Visible text"},
        ],
    }
    res = parser.parse(raw_response)
    assert res.text == "Visible text"
    assert res.thinking == "Internal thought"
    assert len(res.thought_blocks) == 2
    assert res.thought_blocks[0] == {"type": "redacted_thinking", "data": "enc_data_blob_xyz"}
    assert res.thought_blocks[1] == {
        "type": "thinking",
        "thinking": "Internal thought",
        "signature": "sig_thought_1",
    }

    # Content as plain string
    res_str = parser.parse({"content": "Simple raw string response"})
    assert res_str.text == "Simple raw string response"


def test_anthropic_parser_null_response() -> None:
    from evidor.agent.providers.anthropic.parser import AnthropicResponseParser

    parser = AnthropicResponseParser(model="claude-3-7-sonnet")
    res = parser.parse(None)
    assert res.text == ""
    assert res.model == "claude-3-7-sonnet"
    assert res.tool_calls == ()
    assert res.usage is None


# --- AnthropicProvider Lifecycle & Client Pool Tests (4.x fixes) ---


def test_anthropic_provider_base_url_and_with_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(content=[MagicMock(type="text", text="ok")])
    mock_anthropic.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(
        model="claude-3-7-sonnet",
        api_key="ant-key",
        base_url="https://anthropic.custom-proxy.internal",
    )
    assert provider.base_url == "https://anthropic.custom-proxy.internal"

    copied = provider.with_model("claude-3-5-haiku")
    assert copied.model == "claude-3-5-haiku"
    assert copied.base_url == "https://anthropic.custom-proxy.internal"

    provider.generate(GenerationRequest(prompt="hello"))
    mock_anthropic.Anthropic.assert_called_once_with(
        api_key="ant-key",
        base_url="https://anthropic.custom-proxy.internal",
    )


@pytest.mark.asyncio
async def test_anthropic_provider_async_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_client.is_closed = False

    async def _async_create(**kwargs: Any) -> Any:
        return MagicMock(content=[MagicMock(type="text", text="async ok")])

    mock_client.messages.create = _async_create
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(
        model="claude-3-7-sonnet",
        api_key="ant-key",
        base_url="https://anthropic.custom-proxy.internal",
    )
    resp = await provider.generate_async(GenerationRequest(prompt="hello"))
    assert resp.text == "async ok"
    mock_anthropic.AsyncAnthropic.assert_called_once_with(
        api_key="ant-key",
        base_url="https://anthropic.custom-proxy.internal",
    )


@pytest.mark.asyncio
async def test_anthropic_provider_async_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = AnthropicProvider(model="claude-3-5")
    with pytest.raises(ImportError, match="Install Anthropic support with: pip install 'evidor\\[anthropic\\]'"):
        await provider.generate_async(GenerationRequest(prompt="test"))


@pytest.mark.asyncio
async def test_anthropic_async_client_reuse_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_anthropic = types.ModuleType("anthropic")
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.close = AsyncMock()

    async def _async_create(**kwargs: Any) -> Any:
        return MagicMock(content=[MagicMock(type="text", text="res")])

    mock_client.messages.create = _async_create
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    provider = AnthropicProvider(model="claude-3-7-sonnet", api_key="key")
    req = GenerationRequest(prompt="test")

    await provider.generate_async(req)
    await provider.generate_async(req)
    assert mock_anthropic.AsyncAnthropic.call_count == 1

    await provider.close_async()
    mock_client.close.assert_awaited_once()












