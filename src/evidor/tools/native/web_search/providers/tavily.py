"""Tavily web search provider."""

from .._base import SearchResult, normalize_results, resolve_api_key
from ..http import HttpClient, UrllibHttpClient


class TavilySearchProvider:
    """Tavily implementation of the web search provider contract."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, *, http_client: HttpClient | None = None) -> None:
        self._api_key = resolve_api_key(api_key, "TAVILY_API_KEY")
        self._http_client = http_client or UrllibHttpClient()

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        response = self._http_client.request(
            "POST",
            self.endpoint,
            json_body={"api_key": self._api_key, "query": query, "max_results": max_results},
        )
        return normalize_results(response.get("results"), snippet_keys=("content", "raw_content"))[:max_results]
