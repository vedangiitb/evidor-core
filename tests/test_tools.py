from typing import Any, Optional, Union, get_args, get_origin
import pytest

from evidor.tools import Tool, tool


def test_tool_decorator_basic() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.__name__ == "add"
    assert add.description == "Add two numbers together."
    assert add.parameters == {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }
    assert add(2, 3) == 5
    assert add.execute(a=10, b=20) == 30


def test_tool_decorator_with_options() -> None:
    @tool(name="custom_multiply", description="Multiply two numbers.")
    def multiply(x: float, y: float = 1.0) -> float:
        return x * y

    assert multiply.name == "custom_multiply"
    assert multiply.description == "Multiply two numbers."
    assert multiply.parameters == {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x"],
    }
    assert multiply(3.0, 4.0) == 12.0
    assert multiply(3.0) == 3.0


def test_tool_docstring_arg_descriptions() -> None:
    @tool
    def fetch_weather(location: str, unit: Optional[str] = "celsius") -> str:
        """Get the current weather for a city.

        Args:
            location: The city to query.
            unit (str): The temperature unit.

        Returns:
            The weather string.
        """
        return f"{location}: 20 degrees {unit}"

    assert fetch_weather.description == "Get the current weather for a city."
    assert fetch_weather.parameters["properties"]["location"]["description"] == "The city to query."
    assert fetch_weather.parameters["properties"]["unit"]["description"] == "The temperature unit."
    assert fetch_weather.parameters["required"] == ["location"]


def test_tool_sphinx_docstring() -> None:
    @tool
    def search(query: str) -> list[str]:
        """Search documentation.
        :param query: Search query terms
        """
        return [query]

    assert search.description == "Search documentation."
    assert search.parameters["properties"]["query"]["description"] == "Search query terms"
    assert search.parameters["properties"]["query"]["type"] == "string"


def test_tool_types_and_defaults() -> None:
    @tool
    def complex_func(
        flag: bool,
        tags: list[str],
        metadata: dict[str, Any],
        untyped=10,
    ) -> str:
        return "ok"

    props = complex_func.parameters["properties"]
    assert props["flag"]["type"] == "boolean"
    assert props["tags"]["type"] == "array"
    assert props["metadata"]["type"] == "object"
    assert props["untyped"]["type"] == "string"  # untyped falls back to string
    assert complex_func.parameters["required"] == ["flag", "tags", "metadata"]


def test_manual_tool_instantiation() -> None:
    custom_tool = Tool(
        name="manual_echo",
        description="Echo input",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        func=lambda msg: f"Echo: {msg}",
    )
    assert custom_tool.name == "manual_echo"
    assert custom_tool(msg="hello") == "Echo: hello"
    assert custom_tool.execute(msg="world") == "Echo: world"


def test_tool_ignores_self_cls_and_varargs() -> None:
    class MyService:
        @tool
        def method(self, x: int, *args: Any, **kwargs: Any) -> int:
            """Do service."""
            return x

    assert isinstance(MyService.method, Tool)
    assert list(MyService.method.parameters["properties"].keys()) == ["x"]
    assert MyService.method.parameters["required"] == ["x"]


def test_tool_multiline_docstring_arg() -> None:
    @tool
    def process_data(
        raw_text: str,
    ) -> str:
        """Process data.

        Args:
            raw_text: The raw textual data to be processed
                which spans across multiple lines
                for detail.
        """
        return raw_text

    expected_desc = "The raw textual data to be processed which spans across multiple lines for detail."
    assert process_data.parameters["properties"]["raw_text"]["description"] == expected_desc


def test_tool_empty_union_and_failed_type_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    from evidor.tools import _python_type_to_json_type

    # Test Union with only NoneType args
    monkeypatch.setattr("evidor.tools.get_origin", lambda _: Union)
    monkeypatch.setattr("evidor.tools.get_args", lambda _: (type(None),))
    assert _python_type_to_json_type(object) == "string"

    # Test get_type_hints exception fallback
    monkeypatch.setattr("evidor.tools.get_origin", get_origin)
    monkeypatch.setattr("evidor.tools.get_args", get_args)
    monkeypatch.setattr("evidor.tools.get_type_hints", lambda f: (_ for _ in ()).throw(NameError("Cannot resolve type")))

    @tool
    def fallback_func(param: int) -> int:
        return param

    # Falls back to param.annotation
    assert fallback_func.parameters["properties"]["param"]["type"] == "integer"
