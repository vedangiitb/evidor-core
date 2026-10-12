"""Native webpage reader tool for fetching and extracting clean text from URLs."""

from __future__ import annotations

import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request
import urllib.request

from ..core.tools import Tool, tool

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_DEFAULT_MAX_CHARS = 15_000
_DEFAULT_TIMEOUT = 15.0


def _clean_html_fallback(html: str) -> str:
    """Fallback HTML text extractor using regular expressions when BeautifulSoup is absent."""
    # Strip scripts, styles, and comments
    cleaned = re.sub(r"(?is)<script.*?</script>", " ", html)
    cleaned = re.sub(r"(?is)<style.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?is)<(?:nav|header|footer|aside|noscript|svg|iframe).*?</(?:nav|header|footer|aside|noscript|svg|iframe)>", " ", cleaned)

    # Convert block elements and line breaks
    cleaned = re.sub(r"(?i)<(?:p|div|h[1-6]|li|tr|br)\b[^>]*>", "\n", cleaned)
    # Strip all remaining tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Decode basic entities
    cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')

    # Collapse blank lines and whitespace
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line)


def clean_html(html: str) -> str:
    """Extract and normalize readable text content from raw HTML markup.

    Uses :mod:`bs4` (BeautifulSoup) when available for structural extraction,
    falling back to regular expression stripping when absent.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "iframe", "button"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    except ImportError:
        return _clean_html_fallback(html)


def fetch_webpage_content(
    url: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    timeout: float = _DEFAULT_TIMEOUT,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> str:
    """Fetch a webpage by URL and return cleaned text content bounded by ``max_chars``."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return f"Error: Invalid URL scheme '{parsed.scheme}'. Only http and https URLs are supported."
    if not parsed.netloc:
        return f"Error: Invalid URL '{url}'. Missing host/domain."

    req = Request(
        url.strip(),
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" in content_type or "image" in content_type or "octet-stream" in content_type:
                return f"Error: Cannot read binary content type '{content_type}' at {url}."

            raw_bytes = response.read(max_chars * 10)  # Read reasonable slice to protect memory
            charset = response.headers.get_content_charset() or "utf-8"
            html = raw_bytes.decode(charset, errors="replace")
    except HTTPError as exc:
        return f"Error reading {url}: HTTP {exc.code} {exc.reason}"
    except URLError as exc:
        return f"Error reading {url}: Connection error ({exc.reason})"
    except TimeoutError:
        return f"Error reading {url}: Request timed out after {timeout:.1f}s"
    except Exception as exc:
        return f"Error reading {url}: {exc}"

    cleaned = clean_html(html)
    if not cleaned.strip():
        return f"Warning: Successfully loaded {url}, but no readable text content was found."

    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + f"\n\n[Content truncated to {max_chars} characters...]"
    return cleaned


@tool
def read_webpage(url: str) -> str:
    """Fetch a webpage from a URL, strip HTML boilerplate, and return its readable text content.

    Use this tool after searching to inspect primary documentation, articles, benchmarks, or web resources in detail.

    Args:
        url: The web URL to fetch and read (must be http or https).
    """
    return fetch_webpage_content(url, max_chars=_DEFAULT_MAX_CHARS)


def read_webpage_tool(
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    timeout: float = _DEFAULT_TIMEOUT,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> Tool:
    """Create a customized :class:`Tool` for reading webpage contents with custom limits.

    Args:
        max_chars: Maximum character length of returned text before truncation (default: 15,000).
        timeout: HTTP request timeout in seconds (default: 15.0).
        user_agent: Custom User-Agent string for outgoing HTTP requests.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be a positive number")

    def _configured_read(url: str) -> str:
        """Fetch a webpage from a URL, strip HTML boilerplate, and return its readable text content."""
        return fetch_webpage_content(
            url,
            max_chars=max_chars,
            timeout=timeout,
            user_agent=user_agent,
        )

    _configured_read.__name__ = "read_webpage"
    _configured_read.__doc__ = (
        "Fetch a webpage from a URL, strip HTML boilerplate, and return its readable text content.\n\n"
        "Args:\n"
        "    url: The web URL to fetch and read (must be http or https)."
    )
    return tool(_configured_read)


__all__ = ["read_webpage", "read_webpage_tool", "clean_html", "fetch_webpage_content"]

