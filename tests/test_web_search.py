from typing import Any

import pytest

from evidor import (
    BraveSearchProvider,
    ExaSearchProvider,
    SearchResult,
    TavilySearchProvider,
    WebSearchProvider,
    web_search,
)
from evidor.tools import Tool


class RecordingHttpClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


@pytest.mark.parametrize(
    ("provider_class", "api_key", "response", "expected", "method"),
    [
        (
            TavilySearchProvider,
            "tavily-key",
            {"results": [{"title": "Tavily", "url": "https://tavily.com", "content": "answer"}]},
            SearchResult("Tavily", "https://tavily.com", "answer"),
            "POST",
        ),
        (
            ExaSearchProvider,
            "exa-key",
            {"results": [{"title": "Exa", "url": "https://exa.ai", "text": "answer"}]},
            SearchResult("Exa", "https://exa.ai", "answer"),
            "POST",
        ),
        (
            BraveSearchProvider,
            "brave-key",
            {"web": {"results": [{"title": "Brave", "url": "https://brave.com", "description": "answer"}]}},
            SearchResult("Brave", "https://brave.com", "answer"),
            "GET",
        ),
    ],
)
def test_bundled_providers_normalize_results(
    provider_class: type[Any], api_key: str, response: dict[str, Any], expected: SearchResult, method: str
) -> None:
    client = RecordingHttpClient(response)
    provider = provider_class(api_key, http_client=client)

    assert provider.search("dependency inversion", max_results=3) == [expected]
    assert client.calls[0]["method"] == method


def test_tavily_request_uses_its_expected_payload() -> None:
    client = RecordingHttpClient({"results": []})
    TavilySearchProvider("key", http_client=client).search("python", max_results=2)

    assert client.calls == [
        {
            "method": "POST",
            "url": "https://api.tavily.com/search",
            "json_body": {"api_key": "key", "query": "python", "max_results": 2},
        }
    ]


def test_exa_and_brave_requests_use_provider_specific_authentication() -> None:
    exa_client = RecordingHttpClient({"results": []})
    brave_client = RecordingHttpClient({"web": {"results": []}})

    ExaSearchProvider("exa-key", http_client=exa_client).search("python", max_results=2)
    BraveSearchProvider("brave-key", http_client=brave_client).search("two words", max_results=2)

    assert exa_client.calls[0]["headers"] == {"x-api-key": "exa-key"}
    assert exa_client.calls[0]["json_body"] == {"query": "python", "numResults": 2, "contents": {"text": True}}
    assert brave_client.calls[0]["headers"] == {"X-Subscription-Token": "brave-key"}
    assert brave_client.calls[0]["url"].endswith("?q=two+words&count=2")


def test_web_search_tool_depends_only_on_provider_protocol() -> None:
    class CustomProvider:
        def search(self, query: str, *, max_results: int) -> list[SearchResult]:
            assert query == "custom"
            assert max_results == 2
            return [SearchResult("Custom", "https://example.com", "result")]

    search = web_search(CustomProvider(), max_results=2)

    assert isinstance(search, Tool)
    assert SearchResult.__module__ == "evidor.models"
    assert isinstance(CustomProvider(), WebSearchProvider)
    assert search.name == "web_search"
    assert search(query="custom") == [
        {"title": "Custom", "url": "https://example.com", "snippet": "result"}
    ]


def test_web_search_validates_configuration_and_query() -> None:
    class EmptyProvider:
        def search(self, query: str, *, max_results: int) -> list[SearchResult]:
            return []

    with pytest.raises(ValueError, match="positive"):
        web_search(EmptyProvider(), max_results=0)
    with pytest.raises(ValueError, match="must not be empty"):
        web_search(EmptyProvider())(query="   ")


@pytest.mark.parametrize(
    ("provider_class", "environment_variable"),
    [
        (TavilySearchProvider, "TAVILY_API_KEY"),
        (ExaSearchProvider, "EXA_API_KEY"),
        (BraveSearchProvider, "BRAVE_SEARCH_API_KEY"),
    ],
)
def test_providers_read_api_keys_from_environment(
    monkeypatch: pytest.MonkeyPatch, provider_class: type[Any], environment_variable: str
) -> None:
    monkeypatch.setenv(environment_variable, "from-environment")
    provider = provider_class(http_client=RecordingHttpClient({"results": []}))

    assert provider is not None
