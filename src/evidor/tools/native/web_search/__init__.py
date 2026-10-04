"""Provider-agnostic web search tool and bundled search provider adapters."""

from ._base import SearchResult, WebSearchError, WebSearchProvider
from .http import HttpClient, UrllibHttpClient
from .providers import BraveSearchProvider, ExaSearchProvider, TavilySearchProvider
from .tool import web_search

__all__ = [
    "BraveSearchProvider",
    "ExaSearchProvider",
    "HttpClient",
    "SearchResult",
    "TavilySearchProvider",
    "UrllibHttpClient",
    "WebSearchError",
    "WebSearchProvider",
    "web_search",
]
