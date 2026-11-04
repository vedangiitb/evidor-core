"""Request payload builder for Anthropic Claude."""

from collections.abc import Sequence
import json
from typing import Any

from evidor.models import GenerationRequest, Message, ThinkingConfig
from ..base import extract_system_prompt, safe_parse_tool_arguments, sort_tools


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

    def build(self, request: GenerationRequest, thinking: Any = None) -> dict[str, Any]:
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

        active_thinking = thinking if thinking is not None else self.thinking
        if active_thinking and isinstance(active_thinking, str):
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": active_thinking}
        elif active_thinking and not isinstance(active_thinking, (bool, int)):
            budget = (
                active_thinking.get("budget_tokens") or active_thinking.get("thinking_budget")
                if isinstance(active_thinking, dict)
                else getattr(active_thinking, "budget_tokens", None) or getattr(active_thinking, "thinking_budget", None)
            )
            effort = (
                active_thinking.get("effort") or active_thinking.get("thinking_level")
                if isinstance(active_thinking, dict)
                else getattr(active_thinking, "effort", None) or getattr(active_thinking, "thinking_level", None)
            )
            raw_summary = (
                active_thinking.get("summary") or active_thinking.get("display") or active_thinking.get("thinking_summaries")
                if isinstance(active_thinking, dict)
                else getattr(active_thinking, "summary", None) or getattr(active_thinking, "display", None) or getattr(active_thinking, "thinking_summaries", None)
            )
            explicit_type = (
                active_thinking.get("type")
                if isinstance(active_thinking, dict)
                else getattr(active_thinking, "type", None)
            )

            display_val: str | None = None
            if isinstance(raw_summary, bool):
                display_val = "summarized" if raw_summary else "omitted"
            elif isinstance(raw_summary, str) and raw_summary.strip():
                cleaned_summary = raw_summary.strip().lower()
                if cleaned_summary in ("none", "omitted", "off", "hidden", "false"):
                    display_val = "omitted"
                elif cleaned_summary in ("summarized", "summary", "auto", "concise", "detailed", "on", "visible", "true"):
                    display_val = "summarized"
                else:
                    display_val = raw_summary.strip()

            if (budget is not None and not isinstance(budget, bool) and budget > 0) or explicit_type == "enabled":
                actual_budget = max(budget or 1024, 1024)
                thinking_dict: dict[str, Any] = {
                    "type": "enabled",
                    "budget_tokens": actual_budget,
                }
                if display_val:
                    thinking_dict["display"] = display_val
                payload["thinking"] = thinking_dict
                if payload["max_tokens"] <= actual_budget:
                    payload["max_tokens"] = actual_budget + 1024
            elif (
                isinstance(effort, str)
                or display_val is not None
                or explicit_type == "adaptive"
                or isinstance(active_thinking, ThinkingConfig)
            ):
                thinking_dict = {"type": "adaptive"}
                if display_val:
                    thinking_dict["display"] = display_val
                payload["thinking"] = thinking_dict
                if isinstance(effort, str) and effort.strip():
                    payload["output_config"] = {"effort": effort.strip()}

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
        if tb is None:
            return None

        block_dict: dict[str, Any] | None = None
        if isinstance(tb, dict):
            block_dict = tb
        elif hasattr(tb, "model_dump") and callable(tb.model_dump):
            dumped = tb.model_dump()
            if isinstance(dumped, dict):
                block_dict = dumped
        elif hasattr(tb, "to_dict") and callable(tb.to_dict):
            dumped = tb.to_dict()
            if isinstance(dumped, dict):
                block_dict = dumped

        def _get(key: str, default: Any = None) -> Any:
            if block_dict is not None:
                return block_dict.get(key, default)
            return getattr(tb, key, default)

        raw_type = _get("type")
        block_type = str(raw_type).strip().lower() if raw_type is not None else None

        if block_type == "redacted_thinking":
            data_val = _get("data", "")
            return {
                "type": "redacted_thinking",
                "data": str(data_val or ""),
            }

        is_thought = (
            block_type in ("thinking", "thought", "reasoning")
            or _get("thinking") is not None
            or _get("thought") is not None
            or _get("thought_signature") is not None
            or _get("thoughtSignature") is not None
            or _get("signature") is not None
        )

        if is_thought:
            thinking_text = (
                _get("thinking")
                or _get("summary")
                or _get("text")
                or ""
            )
            sig = (
                _get("signature")
                or _get("thought_signature")
                or _get("thoughtSignature")
            )
            normalized: dict[str, Any] = {
                "type": "thinking",
                "thinking": str(thinking_text) if thinking_text is not None else "",
            }
            if sig is not None and str(sig).strip():
                normalized["signature"] = str(sig).strip()
            return normalized

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
                            tc_args = (
                                tc.arguments
                                if isinstance(tc.arguments, dict)
                                else (safe_parse_tool_arguments(tc.arguments) if tc.arguments else {})
                            )
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc_args,
                            })
                    if not content_blocks:
                        fallback_text = message.content if (isinstance(message.content, str) and message.content.strip()) else "Done."
                        content_blocks.append({"type": "text", "text": fallback_text})
                    formatted.append({"role": "assistant", "content": content_blocks})
                else:
                    if isinstance(message.content, list):
                        formatted.append({"role": "assistant", "content": message.content})
                    else:
                        content_val = message.content if (isinstance(message.content, str) and message.content.strip()) else "Done."
                        formatted.append({"role": "assistant", "content": content_val})

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
