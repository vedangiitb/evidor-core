"""Exa web search provider."""

from .._base import SearchResult, normalize_results, resolve_api_key
from ..http import HttpClient, UrllibHttpClient


class ExaSearchProvider:
    """Exa implementation of the web search provider contract."""

    endpoint = "https://api.exa.ai/search"

    def __init__(self, api_key: str | None = None, *, http_client: HttpClient | None = None) -> None:
        self._api_key = resolve_api_key(api_key, "EXA_API_KEY")
        self._http_client = http_client or UrllibHttpClient()

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        response = self._http_client.request(
            "POST",
            self.endpoint,
            headers={"x-api-key": self._api_key},
            json_body={"query": query, "numResults": max_results, "contents": {"text": True}},
        )
        return normalize_results(response.get("results"), snippet_keys=("text", "highlights"))[:max_results]
