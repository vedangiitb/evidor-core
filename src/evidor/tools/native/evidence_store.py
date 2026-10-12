"""In-memory structured evidence and observation store for autonomous agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.tools import Tool, tool


@dataclass
class EvidenceItem:
    """A structured factual claim, metric, trade-off, or observation recorded by an agent."""

    id: str
    topic: str
    finding: str
    source: str = ""
    aspect: str = ""
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def claim(self) -> str:
        """Alias for :attr:`finding`."""
        return self.finding

    @property
    def source_url(self) -> str:
        """Alias for :attr:`source`."""
        return self.source

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence item to a serializable dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "finding": self.finding,
            "source": self.source,
            "aspect": self.aspect,
            "category": self.category,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        parts = [f"[{self.id}] ({self.topic})"]
        if self.category:
            parts.append(f"[{self.category}]")
        parts.append(f": {self.finding}")
        if self.source:
            parts.append(f" (Source: {self.source})")
        return "".join(parts)


@dataclass
class ComparisonItem:
    """A structured comparison across two subjects or conflicting claims on a specific aspect."""

    id: str
    aspect: str
    subject_a: str
    claim_a: str
    subject_b: str
    claim_b: str
    comparison_type: str = "trade-off"
    analysis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def topic_aspect(self) -> str:
        """Alias for :attr:`aspect`."""
        return self.aspect

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison item to a serializable dictionary."""
        return {
            "id": self.id,
            "aspect": self.aspect,
            "subject_a": self.subject_a,
            "claim_a": self.claim_a,
            "subject_b": self.subject_b,
            "claim_b": self.claim_b,
            "comparison_type": self.comparison_type,
            "analysis": self.analysis,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return (
            f"[{self.id}] {self.aspect} ({self.comparison_type}): "
            f"'{self.subject_a}' ({self.claim_a}) vs '{self.subject_b}' ({self.claim_b})"
        )


class EvidenceStore:
    """Accumulates and queries structured evidence, findings, and trade-off comparisons.

    Keeps agent findings structured in memory to prevent context dilution, duplicate
    queries, and hallucinations across long-running tasks.
    """

    def __init__(self, *, prefix: str = "E", comparison_prefix: str = "C") -> None:
        self.prefix = prefix
        self.comparison_prefix = comparison_prefix
        self._items: list[EvidenceItem] = []
        self._comparisons: list[ComparisonItem] = []

    @property
    def items(self) -> list[EvidenceItem]:
        """List of all recorded evidence items."""
        return list(self._items)

    @property
    def evidence(self) -> list[EvidenceItem]:
        """Convenience alias for :attr:`items`."""
        return list(self._items)

    @property
    def comparisons(self) -> list[ComparisonItem]:
        """List of all recorded cross-source or trade-off comparisons."""
        return list(self._comparisons)

    def record(
        self,
        topic: str,
        finding: str,
        *,
        source: str = "",
        aspect: str = "",
        category: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        """Add a new evidence item to the store."""
        item_id = f"{self.prefix}{len(self._items) + 1}"
        item = EvidenceItem(
            id=item_id,
            topic=topic.strip(),
            finding=finding.strip(),
            source=source.strip(),
            aspect=aspect.strip(),
            category=category.strip(),
            metadata=metadata or {},
        )
        self._items.append(item)
        return item

    def record_comparison(
        self,
        aspect: str,
        subject_a: str,
        claim_a: str,
        subject_b: str,
        claim_b: str,
        *,
        comparison_type: str = "trade-off",
        analysis: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ComparisonItem:
        """Add a new comparative claim or trade-off evaluation to the store."""
        comp_id = f"{self.comparison_prefix}{len(self._comparisons) + 1}"
        comp = ComparisonItem(
            id=comp_id,
            aspect=aspect.strip(),
            subject_a=subject_a.strip(),
            claim_a=claim_a.strip(),
            subject_b=subject_b.strip(),
            claim_b=claim_b.strip(),
            comparison_type=comparison_type.strip(),
            analysis=analysis.strip(),
            metadata=metadata or {},
        )
        self._comparisons.append(comp)
        return comp

    def get_by_topic(self, topic: str) -> list[EvidenceItem]:
        """Return all items matching a topic (case-insensitive substring match)."""
        lower = topic.strip().lower()
        return [item for item in self._items if lower in item.topic.lower()]

    def get_by_category(self, category: str) -> list[EvidenceItem]:
        """Return all items matching a category (case-insensitive exact match)."""
        lower = category.strip().lower()
        return [item for item in self._items if item.category.lower() == lower]

    def get_by_id(self, item_id: str) -> EvidenceItem | ComparisonItem | None:
        """Retrieve an item or comparison by its identifier."""
        target = item_id.strip()
        for item in self._items:
            if item.id == target:
                return item
        for comp in self._comparisons:
            if comp.id == target:
                return comp
        return None

    def clear(self) -> None:
        """Reset all recorded evidence and comparisons."""
        self._items.clear()
        self._comparisons.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export all recorded items and comparisons as a JSON-serializable dictionary."""
        return {
            "evidence": [item.to_dict() for item in self._items],
            "comparisons": [comp.to_dict() for comp in self._comparisons],
        }

    def to_markdown(self) -> str:
        """Format all stored evidence and comparisons as structured Markdown."""
        lines = ["# Evidence & Findings", ""]

        if not self._items:
            lines.append("No evidence items recorded.")
        else:
            lines.append("## Findings")
            for item in self._items:
                header = f"- **[{item.id}]** ({item.topic}"
                if item.category:
                    header += f", category: {item.category}"
                header += ")"
                lines.append(header)
                lines.append(f"  - **Finding**: {item.finding}")
                if item.source:
                    lines.append(f"  - **Source**: {item.source}")
                if item.aspect:
                    lines.append(f"  - **Aspect**: {item.aspect}")
            lines.append("")

        if self._comparisons:
            lines.append("## Comparisons & Trade-offs")
            for comp in self._comparisons:
                lines.append(f"- **[{comp.id}]** Aspect: *{comp.aspect}* (Type: {comp.comparison_type})")
                lines.append(f"  - **{comp.subject_a}**: {comp.claim_a}")
                lines.append(f"  - **{comp.subject_b}**: {comp.claim_b}")
                if comp.analysis:
                    lines.append(f"  - **Analysis**: {comp.analysis}")
            lines.append("")

        return "\n".join(lines).strip()

    @property
    def tools(self) -> list[Tool]:
        """List of tools bound to this store, ready to be passed to an Agent."""
        return evidence_store_tools(self)


def evidence_store_tools(store: EvidenceStore) -> list[Tool]:
    """Generate agent tools bound to an :class:`EvidenceStore` instance.

    Returns:
        A list of three tools: ``record_evidence``, ``list_evidence``, and ``compare_evidence``.
    """

    @tool(
        name="record_evidence",
        description="Record a key finding, metric, citation, trade-off, or factual observation into the structured evidence store.",
    )
    def record_evidence(
        topic: str,
        finding: str,
        source: str = "",
        aspect: str = "",
        category: str = "",
    ) -> str:
        """Record a structured evidence item.

        Args:
            topic: The subject, module, framework, or topic area (e.g. 'PostgreSQL', 'Model Latency').
            finding: The factual claim, metric, finding, or observed result.
            source: Source URL, document, or citation origin.
            aspect: The dimension or feature being evaluated (e.g. 'memory', 'throughput', 'architecture').
            category: High-level classification (e.g. 'trade-off', 'benchmark', 'limitation', 'consensus').
        """
        if not topic.strip():
            return "Error: 'topic' cannot be empty."
        if not finding.strip():
            return "Error: 'finding' cannot be empty."

        item = store.record(
            topic=topic,
            finding=finding,
            source=source,
            aspect=aspect,
            category=category,
        )
        return f"Successfully recorded evidence [{item.id}] under topic '{item.topic}'."

    @tool(
        name="list_evidence",
        description="Query and view previously recorded evidence items and findings, with optional topic/aspect/category filters.",
    )
    def list_evidence(
        topic: str = "",
        aspect: str = "",
        category: str = "",
        limit: int = 20,
    ) -> str:
        """List recorded evidence items matching optional filters.

        Args:
            topic: Filter by topic substring (optional).
            aspect: Filter by aspect substring (optional).
            category: Filter by category substring (optional).
            limit: Maximum number of items to return (default: 20).
        """
        results = store.items

        if topic.strip():
            target = topic.strip().lower()
            results = [r for r in results if target in r.topic.lower()]

        if aspect.strip():
            target = aspect.strip().lower()
            results = [r for r in results if target in r.aspect.lower()]

        if category.strip():
            target = category.strip().lower()
            results = [r for r in results if target in r.category.lower()]

        if not results:
            return "No recorded evidence matches the specified criteria."

        selected = results[:limit]
        lines = [f"Found {len(results)} evidence item(s) (showing {len(selected)}):"]
        for item in selected:
            details = [f"[{item.id}] Topic: {item.topic}"]
            if item.category:
                details.append(f"Category: {item.category}")
            if item.aspect:
                details.append(f"Aspect: {item.aspect}")
            lines.append(" | ".join(details))
            lines.append(f"  Finding: {item.finding}")
            if item.source:
                lines.append(f"  Source: {item.source}")

        return "\n".join(lines)

    @tool(
        name="compare_evidence",
        description="Record a cross-comparison between two subjects or claims regarding a specific aspect or trade-off.",
    )
    def compare_evidence(
        aspect: str,
        subject_a: str,
        claim_a: str,
        subject_b: str,
        claim_b: str,
        comparison_type: str = "trade-off",
        analysis: str = "",
    ) -> str:
        """Record a structured comparison between two subjects or claims.

        Args:
            aspect: The specific dimension being compared (e.g. 'Latency', 'Reliability', 'Context Length').
            subject_a: First subject or system name (e.g. 'FastAPI', 'Redis').
            claim_a: Finding or property of first subject.
            subject_b: Second subject or system name (e.g. 'Django', 'Memcached').
            claim_b: Finding or property of second subject.
            comparison_type: Comparison nature, e.g. 'trade-off', 'agreement', 'discrepancy', 'nuance'.
            analysis: Analytical synthesis or conclusion.
        """
        if not aspect.strip():
            return "Error: 'aspect' cannot be empty."
        if not subject_a.strip() or not subject_b.strip():
            return "Error: 'subject_a' and 'subject_b' cannot be empty."

        comp = store.record_comparison(
            aspect=aspect,
            subject_a=subject_a,
            claim_a=claim_a,
            subject_b=subject_b,
            claim_b=claim_b,
            comparison_type=comparison_type,
            analysis=analysis,
        )
        return (
            f"Successfully recorded comparison [{comp.id}] on aspect '{comp.aspect}' "
            f"({comp.comparison_type}): '{comp.subject_a}' vs '{comp.subject_b}'."
        )

    return [record_evidence, list_evidence, compare_evidence]


__all__ = [
    "EvidenceItem",
    "ComparisonItem",
    "EvidenceStore",
    "evidence_store_tools",
]

