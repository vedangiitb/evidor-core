"""Factory for read-only tools scoped to one filesystem root."""

from pathlib import Path

from ...core.tools import Tool
from .list_files import build_list_files
from .read_file import build_read_file
from .search_files import build_search_files


def filesystem_tools(
    root_dir: str | Path,
    *,
    max_file_bytes: int = 100_000,
    max_results: int = 100,
    max_files_scanned: int = 1_000,
) -> tuple[Tool, Tool, Tool]:
    """Create read-only filesystem tools scoped to ``root_dir``.

    File reads are bounded by ``max_file_bytes``. Listing and search output are
    bounded by ``max_results``; recursive searches inspect at most
    ``max_files_scanned`` files.
    """
    root = Path(root_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root_dir must be an existing directory")
    if max_file_bytes < 1 or max_results < 1 or max_files_scanned < 1:
        raise ValueError("Filesystem tool limits must be positive integers")
    return (
        build_list_files(root, max_results),
        build_read_file(root, max_file_bytes),
        build_search_files(root, max_results, max_file_bytes, max_files_scanned),
    )


__all__ = ["filesystem_tools"]
