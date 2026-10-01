from pathlib import Path

import pytest

from evidor import filesystem_tools
from evidor.tools import Tool


def _tools_for(root: Path, **limits: int) -> dict[str, Tool]:
    return {item.name: item for item in filesystem_tools(root, **limits)}


def test_filesystem_tools_are_scoped_to_root_and_exposed_as_tools(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_bytes(b"Alpha project\nsecond line\n")
    nested = root / "docs"
    nested.mkdir()
    (nested / "readme.md").write_bytes(b"alpha appears here\n")
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


def test_filesystem_tools_create_write_and_delete_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "existing.txt").write_text("before", encoding="utf-8")
    tools = _tools_for(root)

    assert tools["create_file"](path="new.txt", content="initial") == "Created new.txt"
    assert (root / "new.txt").read_text(encoding="utf-8") == "initial"
    with pytest.raises(ValueError, match="already exists"):
        tools["create_file"](path="new.txt", content="replace")

    assert tools["write_file"](path="existing.txt", content="after") == "Wrote existing.txt"
    assert (root / "existing.txt").read_text(encoding="utf-8") == "after"
    assert tools["delete_file"](path="existing.txt") == "Deleted existing.txt"
    assert not (root / "existing.txt").exists()


def test_filesystem_mutations_enforce_root_and_write_size_limits(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "outside.txt").write_bytes(b"out")
    (root / "small.txt").write_bytes(b"ok")
    tools = _tools_for(root, max_file_bytes=3)

    with pytest.raises(ValueError, match="outside"):
        tools["create_file"](path="../outside.txt", content="x")
    with pytest.raises(ValueError, match="outside"):
        tools["write_file"](path="../outside.txt", content="x")
    with pytest.raises(ValueError, match="outside"):
        tools["delete_file"](path="../outside.txt")
    with pytest.raises(ValueError, match="write limit"):
        tools["create_file"](path="too-large.txt", content="four")
    with pytest.raises(ValueError, match="write limit"):
        tools["write_file"](path="small.txt", content="four")
    assert (root / "small.txt").read_bytes() == b"ok"


def test_filesystem_mutations_reject_missing_parents_and_directories(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tools = _tools_for(root)

    with pytest.raises(ValueError, match="Parent directory does not exist"):
        tools["create_file"](path="missing/new.txt")
    with pytest.raises(ValueError, match="Not a file"):
        tools["write_file"](path=".", content="text")
    with pytest.raises(ValueError, match="Not a file"):
        tools["delete_file"](path=".")
