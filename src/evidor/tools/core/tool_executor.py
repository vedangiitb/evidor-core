"""Tool registration and provider-requested tool execution."""

from collections.abc import Callable, Sequence
from typing import Any

from ...models import ToolCall
from .tools import Tool, tool
from .utils import format_tool_error, serialize_tool_result


class ToolExecutor:
    """Normalizes configured tools and executes calls requested by a provider."""

    def __init__(self, tools: Sequence[Any] | None, default_timeout: float | None) -> None:
        configured_tools: list[Tool] = []
        for candidate in tools or ():
            if hasattr(candidate, "get_tools") and callable(candidate.get_tools):
                candidates = candidate.get_tools()
            elif hasattr(candidate, "get_evidor_tools") and callable(candidate.get_evidor_tools):
                candidates = candidate.get_evidor_tools()
            else:
                candidates = [candidate]

            for item in candidates:
                configured = item if isinstance(item, Tool) else tool(item)
                if default_timeout is not None and configured.timeout is None:
                    configured = configured.with_timeout(default_timeout)
                configured_tools.append(configured)
        self.tools = tuple(configured_tools)
        self._by_name = {configured.name: configured for configured in self.tools}

    def execute(self, call: ToolCall) -> str:
        target = self._resolve(call)
        if isinstance(target, str):
            return target
        try:
            return serialize_tool_result(target.execute(**call.arguments))
        except Exception as err:
            return format_tool_error(call.name, err)

    async def execute_async(self, call: ToolCall) -> str:
        target = self._resolve(call)
        if isinstance(target, str):
            return target
        try:
            return serialize_tool_result(await target.execute_async(**call.arguments))
        except Exception as err:
            return format_tool_error(call.name, err)

    def _resolve(self, call: ToolCall) -> Tool | str:
        if "__decode_error__" in call.arguments:
            decode_error = call.arguments["__decode_error__"]
            raw_input = call.arguments.get("__raw_args__", "")
            return f"Error: Malformed JSON arguments for tool '{call.name}': {decode_error}. Received input: {raw_input}"
        target = self._by_name.get(call.name)
        return target if target is not None else f"Error: Tool '{call.name}' not found."
