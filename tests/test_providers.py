import sys
import types
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
    mock_openai.OpenAI.assert_called_once_with(api_key="key", max_retries=0)
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

    new_provider = provider.with_model("claude-3-haiku")
    assert new_provider.model == "claude-3-haiku"
    assert new_provider._api_key == "ant-key"
    assert new_provider._max_tokens == 2048


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
    mock_anthropic.Anthropic.assert_called_once_with(api_key="key", max_retries=0)
    mock_client.messages.create.assert_called_once_with(
        model="claude-3-5",
        max_tokens=500,
        system="sys instruction",
        messages=[{"role": "user", "content": "hello"}],
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

    mock_types.Content = MagicMock(side_effect=lambda role, parts: {"role": role, "parts": parts})  # type: ignore[attr-defined]
    mock_types.Part = MagicMock()  # type: ignore[attr-defined]
    mock_types.Part.from_text = MagicMock(side_effect=lambda text: f"part:{text}")
    mock_types.GenerateContentConfig = MagicMock(side_effect=lambda system_instruction: f"config:{system_instruction}")  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_response = MagicMock(text="Gemini generated response")
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash", api_key="key")
    request = GenerationRequest(
        messages=[
            Message(role="system", content="gemini system"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="prior reply"),
        ]
    )

    response = provider.generate(request)

    assert response == GenerationResponse(text="Gemini generated response", model="gemini-2.5-flash")
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents=[
            {"role": "user", "parts": ["part:hello"]},
            {"role": "model", "parts": ["part:prior reply"]},
        ],
        config="config:gemini system",
    )


def test_gemini_provider_generate_no_system_and_none_text(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_types.Content = MagicMock(side_effect=lambda role, parts: {"role": role, "parts": parts})  # type: ignore[attr-defined]
    mock_types.Part = MagicMock()  # type: ignore[attr-defined]
    mock_types.Part.from_text = MagicMock(side_effect=lambda text: f"part:{text}")
    mock_types.GenerateContentConfig = MagicMock()  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=None)
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash")
    response = provider.generate(GenerationRequest(prompt="hello"))

    assert response == GenerationResponse(text="", model="gemini-2.5-flash")
    assert mock_client.models.generate_content.call_args.kwargs["config"] is None


def test_gemini_provider_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    provider = GeminiProvider(model="gemini-2.5-flash")
    with pytest.raises(ImportError, match="Install Gemini support with: pip install 'evidor\\[gemini\\]'"):
        provider.generate(GenerationRequest(prompt="test"))


# --- Provider Tool Calling Tests ---


def test_openai_provider_chat_completions_with_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai = types.ModuleType("openai")
    # Client without responses attribute -> triggers chat.completions fallback
    mock_client = MagicMock(spec=["chat"])
    mock_tc = MagicMock()
    mock_tc.id = "call_abc"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = '{"location": "Tokyo"}'

    mock_choice_msg = MagicMock(content="Checking weather", tool_calls=[mock_tc])
    mock_choice = MagicMock(message=mock_choice_msg)
    mock_response = MagicMock(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
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
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0] == ToolCall(id="call_abc", name="get_weather", arguments={"location": "Tokyo"})

    call_args = mock_client.chat.completions.create.call_args.kwargs
    assert call_args["model"] == "gpt-4o"
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["function"]["name"] == "get_weather"
    assert call_args["messages"][1]["tool_calls"][0]["id"] == "call_abc"
    assert call_args["messages"][2] == {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"}


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

    mock_types.Content = MagicMock(side_effect=lambda role, parts: {"role": role, "parts": parts})  # type: ignore[attr-defined]
    mock_types.Part = MagicMock()  # type: ignore[attr-defined]
    mock_types.Part.from_text = MagicMock(side_effect=lambda text: f"part:{text}")
    mock_types.Part.from_function_response = MagicMock(side_effect=lambda name, response: f"resp:{name}:{response}")
    mock_types.Part.from_function_call = MagicMock(side_effect=lambda name, args: f"fc:{name}:{args}")
    mock_types.FunctionDeclaration = MagicMock(side_effect=lambda name, description, parameters: f"fd:{name}")  # type: ignore[attr-defined]
    mock_types.Tool = MagicMock(side_effect=lambda function_declarations: f"tool:{function_declarations}")  # type: ignore[attr-defined]
    mock_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)  # type: ignore[attr-defined]

    mock_fc = MagicMock()
    mock_fc.id = "call_gemini_1"
    mock_fc.name = "lookup"
    mock_fc.args = {"id": 42}

    mock_response = MagicMock(text="Looking it up", function_calls=[mock_fc])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash")
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
            Message(role="tool", content="data", name="lookup"),
        ],
        tools=(tool_def,),
    )
    resp = provider.generate(request)
    assert resp.text == "Looking it up"
    assert resp.tool_calls == (ToolCall(id="call_gemini_1", name="lookup", arguments={"id": 42}),)


def test_gemini_provider_function_call_via_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_types = types.ModuleType("google.genai.types")

    mock_types.Content = MagicMock(side_effect=lambda role, parts: {"role": role, "parts": parts})  # type: ignore[attr-defined]
    mock_types.Part = MagicMock()  # type: ignore[attr-defined]
    mock_types.Part.from_text = MagicMock(side_effect=lambda text: f"part:{text}")
    mock_types.GenerateContentConfig = MagicMock()  # type: ignore[attr-defined]

    mock_fc = MagicMock()
    mock_fc.id = "call_gemini_2"
    mock_fc.name = "query"
    mock_fc.args = {"term": "python"}

    mock_part = MagicMock(function_call=mock_fc)
    mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]))
    mock_response = MagicMock(text=None, function_calls=None, candidates=[mock_candidate])

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    mock_genai.types = mock_types  # type: ignore[attr-defined]
    mock_google.genai = mock_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    provider = GeminiProvider(model="gemini-2.5-flash")
    resp = provider.generate(GenerationRequest(prompt="query python"))
    assert resp.text == ""
    assert resp.tool_calls == (ToolCall(id="call_gemini_2", name="query", arguments={"term": "python"}),)


def test_openai_provider_handles_malformed_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.agent.providers.openai import _safe_parse_tool_arguments

    assert _safe_parse_tool_arguments(None) == {}
    assert _safe_parse_tool_arguments({"already": "dict"}) == {"already": "dict"}
    assert _safe_parse_tool_arguments('{"valid": 123}') == {"valid": 123}
    assert _safe_parse_tool_arguments('"string"') == {"__raw_args__": "string"}
    
    malformed = _safe_parse_tool_arguments("{bad json")
    assert "__decode_error__" in malformed
    assert malformed["__raw_args__"] == "{bad json"

    # Test through chat.completions.create
    mock_openai = types.ModuleType("openai")
    mock_client = MagicMock(spec=["chat"])
    mock_tc = MagicMock()
    mock_tc.id = "c_bad"
    mock_tc.function.name = "broken"
    mock_tc.function.arguments = "{unclosed json"

    mock_choice = MagicMock(message=MagicMock(content="test", tool_calls=[mock_tc]))
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    mock_openai.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    provider = OpenAIProvider(model="gpt-4o")
    resp = provider.generate(GenerationRequest(prompt="call broken"))
    assert len(resp.tool_calls) == 1
    assert "__decode_error__" in resp.tool_calls[0].arguments
    assert resp.tool_calls[0].arguments["__raw_args__"] == "{unclosed json"

