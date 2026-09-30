from pathlib import Path

import pytest

from evidor import filesystem_tools
from evidor.tools import Tool


def _tools_for(root: Path, **limits: int) -> dict[str, Tool]:
    return {item.name: item for item in filesystem_tools(root, **limits)}


def test_filesystem_tools_are_scoped_to_root_and_exposed_as_tools(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_text("Alpha project\nsecond line\n", encoding="utf-8")
    nested = root / "docs"
    nested.mkdir()
    (nested / "readme.md").write_text("alpha appears here\n", encoding="utf-8")
    tools = _tools_for(root)

    assert all(isinstance(item, Tool) for item in tools.values())
    assert tools["list_files"](path=".") == ["notes.txt"]
    assert tools["list_files"](path=".", recursive=True) == ["notes.txt", "docs/readme.md"]
    assert tools["read_file"](path="notes.txt") == "Alpha project\nsecond line\n"
    assert tools["search_files"](query="ALPHA") == [
        {"path": "notes.txt", "line": 1, "text": "Alpha project"},
        {"path": "docs/readme.md", "line": 1, "text": "alpha appears here"},
    ]


def test_filesystem_tools_reject_paths_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tools = _tools_for(root)

    with pytest.raises(ValueError, match="outside"):
        tools["read_file"](path="../secret.txt")
    with pytest.raises(ValueError, match="outside"):
        tools["list_files"](path="..")


def test_filesystem_tools_enforce_size_and_result_limits(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("hit one\nhit two\n", encoding="utf-8")
    (root / "b.txt").write_text("hit three\n", encoding="utf-8")
    tools = _tools_for(root, max_file_bytes=5, max_results=1)

    assert tools["list_files"](limit=10) == ["a.txt"]
    assert tools["search_files"](query="hit", limit=10) == []
    with pytest.raises(ValueError, match="read limit"):
        tools["read_file"](path="a.txt")


def test_filesystem_tools_require_existing_directory_and_positive_limits(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        filesystem_tools(tmp_path / "missing")
    with pytest.raises(ValueError, match="positive"):
        filesystem_tools(tmp_path, max_results=0)
