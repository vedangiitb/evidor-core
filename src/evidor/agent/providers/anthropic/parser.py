"""Response parser for Anthropic Claude."""

from typing import Any

from evidor.models import GenerationResponse, ToolCall
from ..base import build_token_usage


class AnthropicResponseParser:
    """Parses raw Anthropic SDK responses into normalized GenerationResponse objects."""

    def __init__(self, model: str) -> None:
        self.model = model

    def parse(self, response_msg: Any) -> GenerationResponse:
        """Parse an Anthropic response object into GenerationResponse."""
        msg_id = getattr(response_msg, "id", None)
        msg_id = msg_id if isinstance(msg_id, str) else None

        text, tool_calls, thought_blocks, thought_parts, compaction_summary = self.parse_content_blocks(
            getattr(response_msg, "content", [])
        )

        thinking_text = "".join(thought_parts) if thought_parts else None
        usage = self.parse_usage(getattr(response_msg, "usage", None))

        return GenerationResponse(
            id=msg_id,
            text=text,
            model=self.model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(thought_blocks),
            compaction=compaction_summary,
        )

    def parse_content_blocks(
        self, content_blocks: list[Any]
    ) -> tuple[str, list[ToolCall], list[dict[str, Any]], list[str], str | None]:
        """Extract text, tools, thoughts, and compaction from response content blocks."""
        text_parts: list[str] = []
        thought_parts: list[str] = []
        thought_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        compaction_summary: str | None = None

        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "compaction":
                c = getattr(block, "content", None) or getattr(block, "text", None)
                if c:
                    compaction_summary = str(c)
            elif block_type == "thinking":
                th_text = getattr(block, "thinking", "")
                thought_parts.append(th_text)
                sig = getattr(block, "signature", None)
                tb: dict[str, Any] = {"type": "thinking", "thinking": th_text}
                if sig:
                    tb["signature"] = sig
                thought_blocks.append(tb)
            elif block_type == "redacted_thinking":
                data = getattr(block, "data", None)
                tb = {"type": "redacted_thinking", "data": data}
                thought_blocks.append(tb)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", f"call_{len(tool_calls)+1}"),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}),
                    )
                )

        return "".join(text_parts), tool_calls, thought_blocks, thought_parts, compaction_summary

    def parse_usage(self, raw_usage: Any) -> Any:
        """Parse token usage statistics from Anthropic raw usage."""
        if raw_usage is None:
            return None

        it = getattr(raw_usage, "input_tokens", None)
        ot = getattr(raw_usage, "output_tokens", None)
        cache_read = getattr(raw_usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(raw_usage, "cache_creation_input_tokens", 0) or 0

        if not (isinstance(it, int) or isinstance(ot, int)):
            return None

        prompt_tokens = (int(it) if isinstance(it, int) else 0) + cache_read
        completion_tokens = int(ot) if isinstance(ot, int) else 0

        return build_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_write,
        )

