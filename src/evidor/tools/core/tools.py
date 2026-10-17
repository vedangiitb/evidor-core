import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Any

from evidor.utils import (
    generate_parameters_schema,
    parse_docstring,
    run_async_in_sync,
    run_with_timeout_async,
)


@dataclass(frozen=True, slots=True)
class Tool:
    """A provider-neutral tool definition backed by an executable callable."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    timeout: float | None = None

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def is_async(self) -> bool:
        """Return True if the underlying callable is a coroutine function."""
        return inspect.iscoroutinefunction(self.func)

    def with_timeout(self, timeout: float | None) -> "Tool":
        """Return a copy of this Tool with an updated execution timeout in seconds."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            func=self.func,
            timeout=timeout,
        )

    def _timeout_message(self) -> str:
        return f"Tool '{self.name}' timed out after {self.timeout}s"

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function synchronously with the provided keyword arguments.

        Synchronously executes both standard and coroutine functions.
        Enforces self.timeout if configured.
        """
        return run_async_in_sync(
            lambda: self.execute_async(**kwargs),
            error_message="Async tool cannot be executed synchronously from a running event loop. Use 'await tool.execute_async()' instead.",
        )

    async def execute_async(self, **kwargs: Any) -> Any:
        """Execute the tool function asynchronously, applying timeout if configured."""
        async def _invoke() -> Any:
            if self.is_async:
                return await self.func(**kwargs)
            return await asyncio.to_thread(self.func, **kwargs)

        return await run_with_timeout_async(_invoke(), self.timeout, self._timeout_message())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying function directly."""
        return self.func(*args, **kwargs)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> Any:
    """Decorator to transform a Python function into an Evidor Tool.

    Can be used as:
        @tool
        def my_func(a: int) -> int: ...

    Or with options:
        @tool(name="custom_name", description="custom description", timeout=5.0)
        def my_func(a: int) -> int: ...
    """
    def decorator(fn: Callable[..., Any]) -> Tool:
        doc_summary, param_docs = parse_docstring(fn.__doc__)
        tool_name = name or fn.__name__
        tool_desc = description or doc_summary or f"Tool {tool_name}"
        params_schema = generate_parameters_schema(fn, param_docs)

        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=params_schema,
            func=fn,
            timeout=timeout,
        )

    if func is not None:
        return decorator(func)
    return decorator
