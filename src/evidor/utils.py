"""Utility helpers for schema generation, docstring parsing, and sync-async execution bridging."""

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import contextvars
import inspect
import json
import re
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


TYPE_MAPPING: dict[Any, str] = {
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


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
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


def python_type_to_json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema primitive type."""
    if annotation in (inspect.Parameter.empty, Any):
        return "string"

    origin = get_origin(annotation)
    if origin is not None:
        # Handle Union (e.g. Optional[T] or T | None)
        if origin in (Union, types.UnionType):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                return python_type_to_json_type(args[0])
            return "string"
        if origin in TYPE_MAPPING:
            return TYPE_MAPPING[origin]

    return TYPE_MAPPING.get(annotation, "string")


def generate_parameters_schema(func: Callable[..., Any], param_docs: dict[str, str]) -> dict[str, Any]:
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
        json_type = python_type_to_json_type(annotation)

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


def run_coroutine_sync(factory: Callable[[], Any]) -> Any:
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


def run_with_timeout(func: Callable[[], Any], timeout: float | None, error_message: str) -> Any:
    """Execute a callable synchronously, enforcing a timeout if configured."""
    if timeout is not None and timeout > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as err:
                raise TimeoutError(error_message) from err

    return func()


async def run_with_timeout_async(coro: Any, timeout: float | None, error_message: str) -> Any:
    """Await a coroutine asynchronously, enforcing a timeout if configured."""
    if timeout is not None and timeout > 0:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as err:
            raise TimeoutError(error_message) from err

    return await coro


def serialize_tool_result(result: Any) -> str:
    """Serialize a tool execution result into a string suitable for LLM consumption."""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list, int, float, bool)):
        return json.dumps(result)
    return str(result)


def format_tool_error(name: str, err: Any) -> str:
    """Format a standardized error string when tool execution fails."""
    return f"Error executing tool '{name}': {err}"


