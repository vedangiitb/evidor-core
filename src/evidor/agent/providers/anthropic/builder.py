"""Request payload builder for Anthropic Claude."""

from collections.abc import Sequence
import json
from typing import Any

from evidor.models import GenerationRequest, Message
from ..base import extract_system_prompt, sort_tools


class AnthropicPayloadBuilder:
    """Builds and formats API request payloads for Anthropic's Messages API."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        enable_prompt_caching: bool = True,
        thinking: Any = None,
        compact_threshold: int | None = None,
        compact_instructions: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.enable_prompt_caching = enable_prompt_caching
        self.thinking = thinking
        self.compact_threshold = compact_threshold
        self.compact_instructions = compact_instructions

    def build(self, request: GenerationRequest) -> dict[str, Any]:
        """Construct the full request dictionary for Anthropic Messages API."""
        system_prompt = extract_system_prompt(request)
        formatted_messages = self.format_messages(request.messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt or None,
            "messages": formatted_messages,
        }

        if self.compact_threshold is not None or self.compact_instructions is not None:
            edit: dict[str, Any] = {"type": "compact_20260112"}
            if self.compact_threshold is not None:
                edit["trigger"] = {"type": "input_tokens", "value": self.compact_threshold}
            if self.compact_instructions is not None:
                edit["instructions"] = self.compact_instructions
            payload["betas"] = ["compact-2026-01-12"]
            payload["context_management"] = {"edits": [edit]}

        if self.thinking and self.thinking.enabled:
            if getattr(self.thinking, "type", "enabled") == "adaptive":
                payload["thinking"] = {"type": "adaptive"}
                if getattr(self.thinking, "effort", None):
                    payload["output_config"] = {"effort": self.thinking.effort}
            else:
                budget = getattr(self.thinking, "budget_tokens", None)
                if budget is None or budget < 1024:
                    budget = 1024
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                # Anthropic Messages API requires max_tokens > budget_tokens
                if payload["max_tokens"] <= budget:
                    payload["max_tokens"] = budget + 1024

        if self.enable_prompt_caching:
            payload["cache_control"] = {"type": "ephemeral"}

        if request.tools:
            sorted_tools = sort_tools(request.tools)
            payload["tools"] = [self._format_tool(t) for t in sorted_tools]

        return payload

    @staticmethod
    def _format_tool(tool: Any) -> dict[str, Any]:
        """Format a tool object or dictionary into an Anthropic tool schema."""
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            schema = tool.get("input_schema") or tool.get("parameters")
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            schema = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None)

        if not isinstance(schema, dict) or not schema:
            schema = {"type": "object", "properties": {}}

        tool_dict: dict[str, Any] = {
            "name": name,
            "description": description or "",
            "input_schema": schema,
        }
        return tool_dict

    @staticmethod
    def _normalize_thought_block(tb: Any) -> dict[str, Any] | None:
        """Convert a thought block (dict, Pydantic model, SDK object) to an Anthropic block dict."""
        if isinstance(tb, dict):
            return dict(tb)
        if hasattr(tb, "model_dump") and callable(tb.model_dump):
            dumped = tb.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(tb, "to_dict") and callable(tb.to_dict):
            dumped = tb.to_dict()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(tb, "type"):
            block: dict[str, Any] = {"type": getattr(tb, "type")}
            if hasattr(tb, "thinking") and getattr(tb, "thinking") is not None:
                block["thinking"] = getattr(tb, "thinking")
            if hasattr(tb, "signature") and getattr(tb, "signature") is not None:
                block["signature"] = getattr(tb, "signature")
            if hasattr(tb, "data") and getattr(tb, "data") is not None:
                block["data"] = getattr(tb, "data")
            return block
        return None

    def format_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Format domain messages into Anthropic-compliant message items."""
        formatted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                tool_use_id = message.tool_call_id or "call_unknown"
                content_val = (
                    message.content
                    if isinstance(message.content, (str, list))
                    else ("" if message.content is None else json.dumps(message.content))
                )
                tool_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content_val,
                }
                if getattr(message, "is_error", False):
                    tool_block["is_error"] = True

                if formatted and formatted[-1]["role"] == "user":
                    prev_content = formatted[-1]["content"]
                    if isinstance(prev_content, list):
                        prev_content.append(tool_block)
                    elif isinstance(prev_content, str):
                        formatted[-1]["content"] = [
                            {"type": "text", "text": prev_content},
                            tool_block,
                        ]
                    else:
                        formatted[-1]["content"] = [tool_block]
                else:
                    formatted.append({
                        "role": "user",
                        "content": [tool_block],
                    })

            elif message.role == "assistant":
                if message.thought_blocks or message.tool_calls or message.compaction:
                    content_blocks: list[dict[str, Any]] = []
                    if message.compaction:
                        content_blocks.append({"type": "compaction", "content": message.compaction})
                    # Replay thought_blocks with cryptographic signatures if present (must precede text/tools)
                    if message.thought_blocks:
                        for tb in message.thought_blocks:
                            normalized = self._normalize_thought_block(tb)
                            if normalized is not None:
                                content_blocks.append(normalized)
                    if message.content:
                        content_blocks.append({"type": "text", "text": message.content})
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            })
                    formatted.append({"role": "assistant", "content": content_blocks})
                else:
                    formatted.append({"role": "assistant", "content": message.content})

            else:
                # User or other role
                if formatted and formatted[-1]["role"] == "user":
                    prev_content = formatted[-1]["content"]
                    if isinstance(prev_content, list):
                        if isinstance(message.content, list):
                            prev_content.extend(message.content)
                        elif message.content:
                            prev_content.append({"type": "text", "text": message.content})
                    elif isinstance(prev_content, str):
                        if isinstance(message.content, str):
                            formatted[-1]["content"] = f"{prev_content}\n\n{message.content}"
                        elif isinstance(message.content, list):
                            formatted[-1]["content"] = [{"type": "text", "text": prev_content}] + list(message.content)
                        elif message.content:
                            formatted[-1]["content"] = [
                                {"type": "text", "text": prev_content},
                                {"type": "text", "text": str(message.content)},
                            ]
                else:
                    formatted.append({"role": message.role, "content": message.content})

        return formatted
