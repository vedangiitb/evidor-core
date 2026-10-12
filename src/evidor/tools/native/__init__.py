"""Built-in tools available for explicit opt-in."""

from .calculator import calculator
from .evidence_store import (
    ComparisonItem,
    EvidenceItem,
    EvidenceStore,
    evidence_store_tools,
)
from .filesystem import filesystem_tools
from .get_current_time import get_current_time
from .read_webpage import (
    clean_html,
    fetch_webpage_content,
    read_webpage,
    read_webpage_tool,
)
from .web_search import (
    BraveSearchProvider,
    ExaSearchProvider,
    SearchResult,
    TavilySearchProvider,
    WebSearchProvider,
    web_search,
)

__all__ = [
    "BraveSearchProvider",
    "calculator",
    "clean_html",
    "ComparisonItem",
    "EvidenceItem",
    "EvidenceStore",
    "evidence_store_tools",
    "ExaSearchProvider",
    "fetch_webpage_content",
    "filesystem_tools",
    "get_current_time",
    "read_webpage",
    "read_webpage_tool",
    "SearchResult",
    "TavilySearchProvider",
    "WebSearchProvider",
    "web_search",
]
