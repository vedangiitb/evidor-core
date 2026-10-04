"""Builders for scoped filesystem mutation tools."""

from pathlib import Path

from ...core.tools import Tool, tool
from ._common import resolve_scoped_destination, resolve_scoped_path


def build_create_file(root: Path, max_file_bytes: int) -> Tool:
    @tool
    def create_file(path: str, content: str = "") -> str:
        """Create a new UTF-8 text file under the configured filesystem root.

        Args:
            path: New file path relative to the configured root.
            content: Initial text content.
        """
        target = resolve_scoped_destination(root, path)
        data = content.encode("utf-8")
        if len(data) > max_file_bytes:
            raise ValueError(f"File exceeds the {max_file_bytes}-byte write limit")
        if not target.parent.is_dir():
            raise ValueError(f"Parent directory does not exist: {path}")
        try:
            with target.open("xb") as stream:
                stream.write(data)
        except FileExistsError as exc:
            raise ValueError(f"File already exists: {path}") from exc
        return f"Created {path}"

    return create_file


def build_write_file(root: Path, max_file_bytes: int) -> Tool:
    @tool
    def write_file(path: str, content: str) -> str:
        """Replace the UTF-8 text content of a file under the configured root.

        Args:
            path: Existing file path relative to the configured root.
            content: New text content.
        """
        target = resolve_scoped_path(root, path)
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")
        data = content.encode("utf-8")
        if len(data) > max_file_bytes:
            raise ValueError(f"File exceeds the {max_file_bytes}-byte write limit")
        target.write_bytes(data)
        return f"Wrote {path}"

    return write_file


def build_delete_file(root: Path) -> Tool:
    @tool
    def delete_file(path: str) -> str:
        """Delete a file under the configured filesystem root.

        Args:
            path: Existing file path relative to the configured root.
        """
        target = resolve_scoped_path(root, path)
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")
        target.unlink()
        return f"Deleted {path}"

    return delete_file
