import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import contextvars
from dataclasses import dataclass
import inspect
import re
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


_TYPE_MAPPING = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    tuple: "array",
    set: "array",
    frozenset: "array",
    Sequence: "array",
    Iterable: "array",
    Mapping: "object",
}


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

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function with the provided keyword arguments.

        Synchronously executes both standard and coroutine functions.
        Enforces self.timeout if configured.
        """
        def _invoke() -> Any:
            if inspect.iscoroutinefunction(self.func):
                return self._run_coroutine_factory(lambda: self.func(**kwargs))
            return self.func(**kwargs)

        if self.timeout is not None and self.timeout > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_invoke)
                try:
                    return future.result(timeout=self.timeout)
                except concurrent.futures.TimeoutError as err:
                    raise TimeoutError(f"Tool '{self.name}' timed out after {self.timeout}s") from err

        return _invoke()

    async def execute_async(self, **kwargs: Any) -> Any:
        """Execute the tool function asynchronously, applying timeout if configured."""
        async def _invoke() -> Any:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            return self.func(**kwargs)

        if self.timeout is not None and self.timeout > 0:
            try:
                return await asyncio.wait_for(_invoke(), timeout=self.timeout)
            except (asyncio.TimeoutError, TimeoutError) as err:
                raise TimeoutError(f"Tool '{self.name}' timed out after {self.timeout}s") from err

        return await _invoke()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying function directly."""
        return self.func(*args, **kwargs)

    @staticmethod
    def _run_coroutine_factory(factory: Callable[[], Any]) -> Any:
        """Run an asyncio coroutine factory in the current thread or a worker thread with context preserved."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            ctx = contextvars.copy_context()

            def _worker() -> Any:
                return ctx.run(asyncio.run, factory())

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_worker).result()

        return asyncio.run(factory())


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Extract the main function description and individual parameter descriptions."""
    if not doc:
        return "", {}

    clean_doc = inspect.cleandoc(doc)
    param_descriptions: dict[str, str] = {}

    # Separate summary from sections like Args:, Parameters:, etc.
    split_match = re.split(r"\n\s*(?:Args|Arguments|Parameters):\s*\n", clean_doc, maxsplit=1)
    summary = split_match[0].strip()

    # Also strip Sphinx-style directives (:param, :return, etc.) from summary if present
    summary = re.split(r"\n\s*:[a-zA-Z_]+", summary, maxsplit=1)[0].strip()

    if len(split_match) > 1:
        args_section = split_match[1]
        # Match lines like: 'param_name (type): description' or 'param_name: description'
        # with potential multi-line continuations
        current_param = None
        current_desc: list[str] = []

        for line in args_section.splitlines():
            # Check for next section like 'Returns:', 'Raises:', etc.
            if re.match(r"^\s*(?:Returns|Raises|Yields|Examples?):\s*$", line):
                break

            param_match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\([^)]*\))?\s*:\s*(.*)$", line)
            if param_match:
                if current_param:
                    param_descriptions[current_param] = " ".join(current_desc).strip()
                current_param = param_match.group(1)
                current_desc = [param_match.group(2).strip()]
            elif current_param and line.startswith("    "):
                current_desc.append(line.strip())

        if current_param:
            param_descriptions[current_param] = " ".join(current_desc).strip()

    # Also parse Sphinx-style :param name: description
    for param_match in re.finditer(r":param\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)", clean_doc):
        param_descriptions[param_match.group(1)] = param_match.group(2).strip()

    return summary, param_descriptions


def _python_type_to_json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema primitive type."""
    if annotation in (inspect.Parameter.empty, Any):
        return "string"

    origin = get_origin(annotation)
    if origin is not None:
        # Handle Union (e.g. Optional[T] or T | None)
        if origin in (Union, types.UnionType):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                return _python_type_to_json_type(args[0])
            return "string"
        if origin in _TYPE_MAPPING:
            return _TYPE_MAPPING[origin]

    return _TYPE_MAPPING.get(annotation, "string")


def _generate_parameters_schema(func: Callable[..., Any], param_docs: dict[str, str]) -> dict[str, Any]:
    """Generate a JSON Schema object describing the function parameters."""
    sig = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = type_hints.get(name, param.annotation)
        json_type = _python_type_to_json_type(annotation)

        prop_schema: dict[str, Any] = {"type": json_type}
        if name in param_docs:
            prop_schema["description"] = param_docs[name]

        properties[name] = prop_schema

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


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
        doc_summary, param_docs = _parse_docstring(fn.__doc__)
        tool_name = name or fn.__name__
        tool_desc = description or doc_summary or f"Tool {tool_name}"
        params_schema = _generate_parameters_schema(fn, param_docs)

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
