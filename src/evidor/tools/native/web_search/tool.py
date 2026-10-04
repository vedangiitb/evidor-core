"""The provider-neutral Evidor web search tool factory."""

from dataclasses import asdict

from ...core.tools import Tool, tool
from ._base import WebSearchProvider


def web_search(provider: WebSearchProvider, *, max_results: int = 5) -> Tool:
    """Create an agent tool backed by any ``WebSearchProvider`` implementation."""
    if max_results < 1:
        raise ValueError("max_results must be a positive integer")

    @tool(name="web_search", description="Search the public web and return relevant sources with titles, URLs, and snippets.")
    def search(query: str) -> list[dict[str, str]]:
        """Search the public web.

        Args:
            query: Specific terms or a question to search for.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        return [asdict(result) for result in provider.search(query, max_results=max_results)]

    return search
