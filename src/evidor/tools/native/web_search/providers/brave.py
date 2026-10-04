"""Brave Search web search provider."""

from urllib.parse import urlencode

from .._base import SearchResult, normalize_results, resolve_api_key
from ..http import HttpClient, UrllibHttpClient


class BraveSearchProvider:
    """Brave Search implementation of the web search provider contract."""

    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, *, http_client: HttpClient | None = None) -> None:
        self._api_key = resolve_api_key(api_key, "BRAVE_SEARCH_API_KEY")
        self._http_client = http_client or UrllibHttpClient()

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        url = f"{self.endpoint}?{urlencode({'q': query, 'count': max_results})}"
        response = self._http_client.request("GET", url, headers={"X-Subscription-Token": self._api_key})
        web = response.get("web")
        items = web.get("results") if isinstance(web, dict) else []
        return normalize_results(items, snippet_keys=("description",))[:max_results]
