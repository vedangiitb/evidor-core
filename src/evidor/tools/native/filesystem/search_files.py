"""Read-only text search tool builder."""

import os
from pathlib import Path

from ...core.tools import Tool, tool
from ._common import bounded_limit, resolve_scoped_path


def build_search_files(root: Path, max_results: int, max_file_bytes: int, max_files_scanned: int) -> Tool:
    @tool
    def search_files(query: str, path: str = ".", recursive: bool = True, limit: int = 50) -> list[dict[str, str | int]]:
        """Search UTF-8 text files under the configured root for a literal string.

        Args:
            query: Text to find, matched without regard to letter case.
            path: File or directory path relative to the configured root.
            recursive: Whether to search nested directories when path is a directory.
            limit: Maximum number of matching lines to return.
        """
        if not query:
            raise ValueError("query must not be empty")
        target = resolve_scoped_path(root, path)
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = []
            if recursive:
                for current, dirnames, filenames in os.walk(target, followlinks=False):
                    dirnames[:] = [name for name in sorted(dirnames) if not (Path(current) / name).is_symlink()]
                    remaining = max_files_scanned - len(candidates)
                    candidates.extend(Path(current) / name for name in sorted(filenames)[:remaining])
                    if len(candidates) >= max_files_scanned:
                        break
            else:
                candidates = [item for item in sorted(target.iterdir(), key=lambda item: item.name) if item.is_file()]
            candidates = candidates[:max_files_scanned]
        else:
            raise ValueError(f"Not a file or directory: {path}")

        result_limit = bounded_limit(limit, max_results)
        matches: list[dict[str, str | int]] = []
        needle = query.casefold()
        for candidate in candidates:
            try:
                candidate = resolve_scoped_path(root, str(candidate))
                if not candidate.is_file() or candidate.stat().st_size > max_file_bytes:
                    continue
                with candidate.open("rb") as stream:
                    content = stream.read(max_file_bytes + 1)
                if len(content) > max_file_bytes:
                    continue
                for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
                    if needle in line.casefold():
                        matches.append({
                            "path": candidate.relative_to(root).as_posix(),
                            "line": line_number,
                            "text": line[:1000],
                        })
                        if len(matches) >= result_limit:
                            return matches
            except (OSError, UnicodeDecodeError, ValueError):
                continue
        return matches

    return search_files
