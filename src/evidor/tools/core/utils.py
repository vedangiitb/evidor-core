"""Utilities used by core tool execution components."""

import json
from typing import Any


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
