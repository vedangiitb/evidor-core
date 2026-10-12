from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request
from unittest.mock import MagicMock, patch
import pytest

from evidor import (
    ComparisonItem,
    EvidenceItem,
    EvidenceStore,
    evidence_store_tools,
    read_webpage,
    read_webpage_tool,
)
from evidor.tools import Tool
from evidor.tools.native.read_webpage import clean_html, _clean_html_fallback, fetch_webpage_content


# ---------------------------------------------------------------------------
# read_webpage tool tests
# ---------------------------------------------------------------------------

def test_read_webpage_is_tool_instance() -> None:
    assert isinstance(read_webpage, Tool)
    assert read_webpage.name == "read_webpage"
    assert "url" in read_webpage.parameters["properties"]


def test_read_webpage_tool_factory_custom_config() -> None:
    custom_tool = read_webpage_tool(max_chars=500, timeout=5.0)
    assert isinstance(custom_tool, Tool)
    assert custom_tool.name == "read_webpage"

    with pytest.raises(ValueError, match="max_chars must be a positive integer"):
        read_webpage_tool(max_chars=0)

    with pytest.raises(ValueError, match="timeout must be a positive number"):
        read_webpage_tool(timeout=-1.0)


def test_read_webpage_invalid_scheme() -> None:
    result = read_webpage(url="ftp://example.com/file.txt")
    assert "Error: Invalid URL scheme 'ftp'" in result

    result_file = read_webpage(url="file:///etc/passwd")
    assert "Error: Invalid URL scheme 'file'" in result_file


def test_clean_html_fallback() -> None:
    html = """
    <html>
      <head><title>Test Title</title><style>.foo { color: red; }</style></head>
      <body>
        <script>console.log("bad");</script>
        <header>Header content</header>
        <nav><a href="/">Home</a></nav>
        <h1>Hello World</h1>
        <p>This is a &lt;paragraph&gt; &amp; test &quot;quote&quot;&nbsp;here.</p>
        <footer>Footer content</footer>
      </body>
    </html>
    """
    text = _clean_html_fallback(html)
    assert "console.log" not in text
    assert ".foo" not in text
    assert "Header content" not in text
    assert "Home" not in text
    assert "Hello World" in text
    assert "This is a <paragraph> & test \"quote\" here." in text


def test_clean_html_extractor() -> None:
    html = "<div><h1>Title</h1><p>Content text</p></div>"
    cleaned = clean_html(html)
    assert "Title" in cleaned
    assert "Content text" in cleaned


def test_fetch_webpage_content_success() -> None:
    fake_html = b"<html><body><h1>Docs</h1><p>Evidor is an agent framework.</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "text/html; charset=utf-8"
    mock_resp.headers.get_content_charset.return_value = "utf-8"
    mock_resp.read.return_value = fake_html
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("evidor.tools.native.read_webpage.urlopen", return_value=mock_resp):
        content = fetch_webpage_content("https://example.com/docs")
        assert "Docs" in content
        assert "Evidor is an agent framework." in content


def test_fetch_webpage_content_truncation() -> None:
    long_text = b"<html><body>" + b"<p>" + (b"word " * 500) + b"</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "text/html; charset=utf-8"
    mock_resp.headers.get_content_charset.return_value = "utf-8"
    mock_resp.read.return_value = long_text
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("evidor.tools.native.read_webpage.urlopen", return_value=mock_resp):
        content = fetch_webpage_content("https://example.com/long", max_chars=100)
        assert len(content) > 100
        assert "[Content truncated to 100 characters...]" in content


def test_fetch_webpage_content_binary_skip() -> None:
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "application/pdf"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("evidor.tools.native.read_webpage.urlopen", return_value=mock_resp):
        content = fetch_webpage_content("https://example.com/manual.pdf")
        assert "Cannot read binary content type 'application/pdf'" in content


def test_fetch_webpage_content_http_error() -> None:
    http_err = HTTPError("https://example.com/404", 404, "Not Found", None, None)  # type: ignore[arg-type]
    with patch("evidor.tools.native.read_webpage.urlopen", side_effect=http_err):
        content = fetch_webpage_content("https://example.com/404")
        assert "Error reading https://example.com/404: HTTP 404 Not Found" in content


def test_fetch_webpage_content_url_error() -> None:
    url_err = URLError("Name or service not known")
    with patch("evidor.tools.native.read_webpage.urlopen", side_effect=url_err):
        content = fetch_webpage_content("https://nonexistent.domain.xyz")
        assert "Connection error (Name or service not known)" in content


# ---------------------------------------------------------------------------
# EvidenceStore tests
# ---------------------------------------------------------------------------

def test_evidence_store_record_and_query() -> None:
    store = EvidenceStore()
    assert len(store.items) == 0

    item1 = store.record(
        topic="PostgreSQL",
        finding="Connection pooling reduces latency by 40% under high concurrency.",
        source="https://postgres.org/docs",
        aspect="performance",
        category="benchmark",
    )

    assert item1.id == "E1"
    assert item1.topic == "PostgreSQL"
    assert item1.aspect == "performance"
    assert "E1" in str(item1)
    assert len(store.items) == 1
    assert store.evidence[0] == item1

    item2 = store.record(
        topic="SQLite",
        finding="Single-writer lock limits concurrency in multi-process setups.",
        aspect="concurrency",
        category="limitation",
    )
    assert item2.id == "E2"
    assert len(store.items) == 2

    # Query by topic
    postgres_items = store.get_by_topic("postgres")
    assert len(postgres_items) == 1
    assert postgres_items[0].id == "E1"

    # Query by category
    limitations = store.get_by_category("limitation")
    assert len(limitations) == 1
    assert limitations[0].id == "E2"

    # Query by ID
    assert store.get_by_id("E1") == item1
    assert store.get_by_id("E99") is None


def test_evidence_store_comparisons() -> None:
    store = EvidenceStore()
    comp = store.record_comparison(
        aspect="Write Concurrency",
        subject_a="PostgreSQL",
        claim_a="MVCC supports high parallel write throughput.",
        subject_b="SQLite",
        claim_b="Database-level write lock serializes write transactions.",
        comparison_type="trade-off",
        analysis="PostgreSQL is better for write-heavy services; SQLite for local or read-heavy.",
    )

    assert comp.id == "C1"
    assert comp.aspect == "Write Concurrency"
    assert len(store.comparisons) == 1
    assert store.get_by_id("C1") == comp
    assert "C1" in str(comp)

    # Dictionary export
    exported = store.to_dict()
    assert "evidence" in exported
    assert len(exported["comparisons"]) == 1
    assert exported["comparisons"][0]["subject_a"] == "PostgreSQL"


def test_evidence_store_markdown_export() -> None:
    store = EvidenceStore()
    assert "No evidence items recorded." in store.to_markdown()

    store.record(topic="Agent Memory", finding="Scratchpad buffers context.")
    store.record_comparison(
        aspect="Memory",
        subject_a="Scratchpad",
        claim_a="Clean tokens",
        subject_b="Raw History",
        claim_b="Pollutes context",
    )

    md = store.to_markdown()
    assert "# Evidence & Findings" in md
    assert "## Findings" in md
    assert "Agent Memory" in md
    assert "## Comparisons & Trade-offs" in md
    assert "Scratchpad" in md


def test_evidence_store_tools_execution() -> None:
    store = EvidenceStore()
    tools = store.tools
    assert len(tools) == 3

    tool_names = {t.name for t in tools}
    assert tool_names == {"record_evidence", "list_evidence", "compare_evidence"}

    record_tool = next(t for t in tools if t.name == "record_evidence")
    list_tool = next(t for t in tools if t.name == "list_evidence")
    compare_tool = next(t for t in tools if t.name == "compare_evidence")

    # Validation errors
    assert "Error: 'topic' cannot be empty." in record_tool(topic="", finding="some finding")
    assert "Error: 'finding' cannot be empty." in record_tool(topic="Topic", finding="")

    # Successful record
    rec_resp = record_tool(
        topic="Redis",
        finding="In-memory sub-millisecond reads.",
        source="https://redis.io",
        aspect="latency",
        category="benchmark",
    )
    assert "Successfully recorded evidence [E1]" in rec_resp
    assert len(store.items) == 1

    # List evidence
    list_resp = list_tool()
    assert "Found 1 evidence item" in list_resp
    assert "Redis" in list_resp

    # Filtered list evidence
    assert "No recorded evidence matches" in list_tool(topic="PostgreSQL")
    assert "Found 1 evidence item" in list_tool(topic="redis")

    # Compare tool
    assert "Error: 'aspect' cannot be empty." in compare_tool(
        aspect="", subject_a="A", claim_a="ca", subject_b="B", claim_b="cb"
    )

    comp_resp = compare_tool(
        aspect="Storage",
        subject_a="Redis",
        claim_a="RAM-first",
        subject_b="Disk",
        claim_b="Slower IO",
        comparison_type="trade-off",
        analysis="Trade RAM cost for latency.",
    )
    assert "Successfully recorded comparison [C1]" in comp_resp
    assert len(store.comparisons) == 1

    # Clear store
    store.clear()
    assert len(store.items) == 0
    assert len(store.comparisons) == 0

