"""Request payload builder for OpenAI Responses API."""

import json
from collections.abc import Sequence
from typing import Any

from evidor.models import GenerationRequest, Message
from ..base import sort_tools


class OpenAIPayloadBuilder:
    """Builds and formats API request payloads for OpenAI Responses API."""

    def __init__(
        self,
        model: str,
        thinking: Any = None,
        compact_threshold: int | None = None,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.compact_threshold = compact_threshold

    def build(self, request: GenerationRequest) -> dict[str, Any]:
        """Construct the full request dictionary for OpenAI Responses API."""
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self.build_input_items(request.messages),
        }

        if request.system:
            payload["instructions"] = request.system
        if request.previous_response_id:
            payload["previous_response_id"] = request.previous_response_id
        if request.conversation_id:
            payload["conversation"] = request.conversation_id
        if request.store is not None:
            payload["store"] = request.store

        if self.compact_threshold is not None:
            payload["context_management"] = [
                {"type": "compaction", "compact_threshold": self.compact_threshold}
            ]

        if self.thinking and self.thinking.enabled:
            reasoning_config: dict[str, Any] = {}
            if self.thinking.effort:
                reasoning_config["effort"] = self.thinking.effort
            summary_val = getattr(self.thinking, "summary", None)
            if summary_val:
                reasoning_config["summary"] = summary_val
            if reasoning_config:
                payload["reasoning"] = reasoning_config

        if request.tools:
            sorted_tools = sort_tools(request.tools)
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in sorted_tools
            ]

        return payload

    def build_input_items(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Format domain messages into OpenAI Responses API input items."""
        input_items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                })
            elif message.tool_calls:
                for tc in message.tool_calls:
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                    })
                if message.content:
                    input_items.append({"role": "assistant", "content": message.content})
            else:
                input_items.append({"role": message.role, "content": message.content})
        return input_items

