"""Shared contracts and result normalization for web search providers."""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from ....models import SearchResult


class WebSearchError(RuntimeError):
    """Raised when a web search provider cannot complete a request."""


@runtime_checkable
class WebSearchProvider(Protocol):
    """Port implemented by services capable of searching the web."""

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Return at most ``max_results`` results for ``query``."""


def resolve_api_key(value: str | None, environment_variable: str) -> str:
    """Resolve an explicit API key or one provided through the environment."""
    key = value or os.getenv(environment_variable)
    if not key:
        raise ValueError(f"{environment_variable} must be provided or set in the environment")
    return key


def normalize_results(items: Any, *, snippet_keys: tuple[str, ...]) -> list[SearchResult]:
    """Convert a provider result list into Evidor's stable result shape."""
    if not isinstance(items, list):
        return []

    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        title = item.get("title")
        snippet = next(
            (item[key] for key in snippet_keys if isinstance(item.get(key), str) and item[key]),
            "",
        )
        results.append(SearchResult(title=title if isinstance(title, str) else "", url=url, snippet=snippet))
    return results
