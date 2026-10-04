"""HTTP boundary used by bundled web search providers."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._base import WebSearchError


class HttpClient(Protocol):
    """Minimal JSON HTTP client contract for search provider adapters."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a JSON request and return a decoded JSON object."""


class UrllibHttpClient:
    """Dependency-free :class:`HttpClient` implementation."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._timeout = timeout

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_headers = {"Accept": "application/json", **(headers or {})}
        data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        if data is not None:
            request_headers.setdefault("Content-Type", "application/json")
        request = Request(url, data=data, headers=request_headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            raise WebSearchError(f"Web search request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise WebSearchError(f"Web search request failed: {exc.reason}") from exc

        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise WebSearchError("Web search provider returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise WebSearchError("Web search provider returned an invalid response")
        return decoded
