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
