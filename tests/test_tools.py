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
    monkeypatch.setattr("evidor.utils.schema.get_origin", lambda _: Union)
    monkeypatch.setattr("evidor.utils.schema.get_args", lambda _: (type(None),))
    assert _python_type_to_json_type(object) == "string"

    # Test get_type_hints exception fallback
    monkeypatch.setattr("evidor.utils.schema.get_origin", get_origin)
    monkeypatch.setattr("evidor.utils.schema.get_args", get_args)
    monkeypatch.setattr(
        "evidor.utils.schema.get_type_hints",
        lambda f: (_ for _ in ()).throw(NameError("Cannot resolve type")),
    )

    @tool
    def fallback_func(param: int) -> int:
        return param

    # Falls back to param.annotation
    assert fallback_func.parameters["properties"]["param"]["type"] == "integer"


def test_async_tool_execution() -> None:
    import asyncio

    @tool
    async def async_add(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        await asyncio.sleep(0.01)
        return a + b

    assert async_add.is_async is True
    # Test sync execution wrapper
    assert async_add.execute(a=3, b=4) == 7

    async def _run() -> None:
        # Test async execution method
        assert await async_add.execute_async(a=10, b=20) == 30
        # Test direct callable invocation
        assert await async_add(5, 5) == 10

        # Test sync function through execute_async
        @tool
        def sync_multiply(x: int, y: int) -> int:
            return x * y

        assert await sync_multiply.execute_async(x=3, y=4) == 12

    asyncio.run(_run())


def test_async_tool_execution_from_sync_context() -> None:
    import asyncio

    @tool
    async def async_greet(name: str) -> str:
        await asyncio.sleep(0.01)
        return f"Hello, {name}"

    assert async_greet.execute(name="World") == "Hello, World"


def test_tool_timeout_sync() -> None:
    import time

    @tool(timeout=0.05)
    def slow_func(delay: float) -> str:
        time.sleep(delay)
        return "done"

    assert slow_func.timeout == 0.05
    with pytest.raises(TimeoutError, match="timed out after 0.05s"):
        slow_func.execute(delay=0.2)

    # Test with_timeout
    fast_func = slow_func.with_timeout(1.0)
    assert fast_func.timeout == 1.0
    assert fast_func.execute(delay=0.01) == "done"


def test_tool_timeout_async() -> None:
    import asyncio

    @tool(timeout=0.05)
    async def slow_async_func(delay: float) -> str:
        await asyncio.sleep(delay)
        return "done"

    async def _run() -> None:
        with pytest.raises(TimeoutError, match="timed out after 0.05s"):
            await slow_async_func.execute_async(delay=0.2)

    asyncio.run(_run())


def test_async_tool_preserves_contextvars() -> None:
    import asyncio
    import contextvars

    test_var: contextvars.ContextVar[str] = contextvars.ContextVar("test_var", default="default")
    test_var.set("custom_token_123")

    @tool
    async def get_token() -> str:
        await asyncio.sleep(0.001)
        return test_var.get()

    async def _run() -> None:
        # When executed synchronously inside an active loop (which dispatches to worker thread with context copy)
        assert get_token.execute() == "custom_token_123"
        # When executed asynchronously in native loop
        assert await get_token.execute_async() == "custom_token_123"

    asyncio.run(_run())
