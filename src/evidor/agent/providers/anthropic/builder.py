"""Request payload builder for Anthropic Claude."""

from collections.abc import Sequence
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
            if self.thinking.type == "adaptive":
                payload["thinking"] = {"type": "adaptive"}
                if self.thinking.effort:
                    payload["output_config"] = {"effort": self.thinking.effort}
            elif self.thinking.budget_tokens:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking.budget_tokens,
                }

        if self.enable_prompt_caching:
            payload["cache_control"] = {"type": "ephemeral"}

        if request.tools:
            sorted_tools = sort_tools(request.tools)
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in sorted_tools
            ]

        return payload

    def format_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Format domain messages into Anthropic-compliant message items."""
        formatted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                })
            elif message.role == "assistant":
                if message.thought_blocks or message.tool_calls or message.compaction:
                    content_blocks: list[dict[str, Any]] = []
                    if message.compaction:
                        content_blocks.append({"type": "compaction", "content": message.compaction})
                    # Replay thought_blocks with cryptographic signatures if present
                    if message.thought_blocks:
                        for tb in message.thought_blocks:
                            if isinstance(tb, dict):
                                content_blocks.append(dict(tb))
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
                formatted.append({"role": message.role, "content": message.content})
        return formatted

