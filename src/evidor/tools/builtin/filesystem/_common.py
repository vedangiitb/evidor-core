"""Shared path and result-limit helpers for filesystem tools."""

from pathlib import Path


def resolve_scoped_path(root: Path, user_path: str) -> Path:
    """Resolve a model-provided path and reject paths outside the configured root."""
    candidate = (root / user_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path is outside the configured filesystem root") from exc
    return candidate


def bounded_limit(requested: int, maximum: int) -> int:
    """Clamp a requested result count to a safe positive upper bound."""
    if requested < 1:
        raise ValueError("limit must be at least 1")
    return min(requested, maximum)
