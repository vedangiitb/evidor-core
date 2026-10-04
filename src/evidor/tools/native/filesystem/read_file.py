"""Read-only file content tool builder."""

from pathlib import Path

from ...core.tools import Tool, tool
from ._common import resolve_scoped_path


def build_read_file(root: Path, max_file_bytes: int) -> Tool:
    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file under the configured filesystem root.

        Args:
            path: File path relative to the configured root.
        """
        target = resolve_scoped_path(root, path)
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")
        if target.stat().st_size > max_file_bytes:
            raise ValueError(f"File exceeds the {max_file_bytes}-byte read limit")
        try:
            with target.open("rb") as stream:
                content = stream.read(max_file_bytes + 1)
            if len(content) > max_file_bytes:
                raise ValueError(f"File exceeds the {max_file_bytes}-byte read limit")
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("File is not valid UTF-8 text") from exc

    return read_file
