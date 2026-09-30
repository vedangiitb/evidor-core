"""Docstring inspection, Python-type conversion, and JSON Schema generation."""

from collections.abc import Callable, Iterable, Mapping, Sequence
import inspect
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
    split_match = re.split(r"\n\s*(?:Args|Arguments|Parameters):\s*\n", clean_doc, maxsplit=1)
    summary = re.split(r"\n\s*:[a-zA-Z_]+", split_match[0].strip(), maxsplit=1)[0].strip()

    if len(split_match) > 1:
        current_param = None
        current_desc: list[str] = []
        for line in split_match[1].splitlines():
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

    for param_match in re.finditer(r":param\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)", clean_doc):
        param_descriptions[param_match.group(1)] = param_match.group(2).strip()
    return summary, param_descriptions


def python_type_to_json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema primitive type."""
    if annotation in (inspect.Parameter.empty, Any):
        return "string"

    origin = get_origin(annotation)
    if origin is not None:
        if origin in (Union, types.UnionType):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            return python_type_to_json_type(args[0]) if args else "string"
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
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        prop_schema: dict[str, Any] = {"type": python_type_to_json_type(type_hints.get(name, param.annotation))}
        if name in param_docs:
            prop_schema["description"] = param_docs[name]
        properties[name] = prop_schema
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}
