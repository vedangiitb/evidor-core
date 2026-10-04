"""Read-only directory listing tool builder."""

import os
from pathlib import Path

from ...core.tools import Tool, tool
from ._common import bounded_limit, resolve_scoped_path


def build_list_files(root: Path, max_results: int) -> Tool:
    @tool
    def list_files(path: str = ".", recursive: bool = False, limit: int = 100) -> list[str]:
        """List file paths in a directory under the configured filesystem root.

        Args:
            path: Directory path relative to the configured root.
            recursive: Whether to include files in nested directories.
            limit: Maximum number of paths to return.
        """
        directory = resolve_scoped_path(root, path)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {path}")
        result_limit = bounded_limit(limit, max_results)
        found: list[str] = []
        if recursive:
            for current, dirnames, filenames in os.walk(directory, followlinks=False):
                dirnames[:] = [name for name in sorted(dirnames) if not (Path(current) / name).is_symlink()]
                for name in sorted(filenames):
                    candidate = Path(current) / name
                    try:
                        resolved = resolve_scoped_path(root, str(candidate))
                    except (OSError, ValueError):
                        continue
                    if resolved.is_file():
                        found.append(resolved.relative_to(root).as_posix())
                        if len(found) >= result_limit:
                            return found
        else:
            for candidate in sorted(directory.iterdir(), key=lambda item: item.name):
                if not candidate.is_file():
                    continue
                try:
                    resolved = resolve_scoped_path(root, str(candidate))
                except (OSError, ValueError):
                    continue
                found.append(resolved.relative_to(root).as_posix())
                if len(found) >= result_limit:
                    break
        return found

    return list_files
