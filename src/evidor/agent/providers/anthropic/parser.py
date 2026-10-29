"""Response parser for Anthropic Claude."""

from typing import Any

from evidor.models import GenerationResponse, ToolCall
from ..base import build_token_usage, safe_parse_tool_arguments


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Retrieve an attribute or dict key safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class AnthropicResponseParser:
    """Parses raw Anthropic SDK responses into normalized GenerationResponse objects."""

    def __init__(self, model: str) -> None:
        self.model = model

    def parse(self, response_msg: Any) -> GenerationResponse:
        """Parse an Anthropic response object or dict into GenerationResponse."""
        if response_msg is None:
            return GenerationResponse(text="", model=self.model)

        raw_id = _get_field(response_msg, "id")
        msg_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else None

        raw_model = _get_field(response_msg, "model")
        resp_model = raw_model if isinstance(raw_model, str) and raw_model.strip() else self.model

        raw_content = _get_field(response_msg, "content", [])
        text, tool_calls, thought_blocks, thought_parts, compaction_summary = self.parse_content_blocks(
            raw_content
        )

        thinking_text = "".join(thought_parts) if thought_parts else None
        if thinking_text is None:
            raw_thinking = _get_field(response_msg, "thinking") or _get_field(response_msg, "reasoning")
            if isinstance(raw_thinking, str) and raw_thinking.strip():
                thinking_text = raw_thinking.strip()

        usage = self.parse_usage(_get_field(response_msg, "usage"))

        return GenerationResponse(
            id=msg_id,
            text=text,
            model=resp_model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(thought_blocks),
            compaction=compaction_summary,
        )

    def parse_content_blocks(
        self, content_blocks: Any
    ) -> tuple[str, list[ToolCall], list[dict[str, Any]], list[str], str | None]:
        """Extract text, tools, thoughts, and compaction from response content blocks."""
        text_parts: list[str] = []
        thought_parts: list[str] = []
        thought_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        compaction_summary: str | None = None

        if not content_blocks:
            return "", [], [], [], None

        if isinstance(content_blocks, str):
            return content_blocks, [], [], [], None

        blocks = content_blocks if isinstance(content_blocks, (list, tuple)) else [content_blocks]

        for block in blocks:
            if isinstance(block, str):
                text_parts.append(block)
                continue

            block_type = _get_field(block, "type")
            if block_type == "text":
                t = _get_field(block, "text", "")
                if t:
                    text_parts.append(str(t))
            elif block_type == "compaction":
                c = (
                    _get_field(block, "content")
                    or _get_field(block, "text")
                    or _get_field(block, "summary")
                    or _get_field(block, "data")
                )
                if c:
                    compaction_summary = str(c)
            elif block_type == "thinking":
                th_text = _get_field(block, "thinking") or _get_field(block, "text") or ""
                thought_parts.append(str(th_text))
                sig = _get_field(block, "signature")
                tb: dict[str, Any] = {"type": "thinking", "thinking": str(th_text)}
                if isinstance(sig, str) and sig.strip():
                    tb["signature"] = sig.strip()
                thought_blocks.append(tb)
            elif block_type == "redacted_thinking":
                data = _get_field(block, "data")
                tb_redacted: dict[str, Any] = {"type": "redacted_thinking"}
                if data is not None and isinstance(data, str):
                    tb_redacted["data"] = data
                elif data is not None:
                    tb_redacted["data"] = str(data)
                thought_blocks.append(tb_redacted)
            elif block_type == "tool_use":
                raw_id = _get_field(block, "id")
                call_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else f"call_{len(tool_calls)+1}"

                raw_name = _get_field(block, "name")
                tool_name = raw_name if isinstance(raw_name, str) and raw_name.strip() else "unknown_tool"

                raw_input = _get_field(block, "input")
                tool_args = safe_parse_tool_arguments(raw_input)

                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=tool_name,
                        arguments=tool_args,
                    )
                )

        return "".join(text_parts), tool_calls, thought_blocks, thought_parts, compaction_summary

    def parse_usage(self, raw_usage: Any) -> Any:
        """Parse token usage statistics from Anthropic raw usage."""
        if raw_usage is None:
            return None

        it = _get_field(raw_usage, "input_tokens")
        ot = _get_field(raw_usage, "output_tokens")
        raw_cache_read = _get_field(raw_usage, "cache_read_input_tokens")
        raw_cache_write = _get_field(raw_usage, "cache_creation_input_tokens")

        cache_read = (
            int(raw_cache_read)
            if isinstance(raw_cache_read, (int, float)) and not isinstance(raw_cache_read, bool)
            else 0
        )
        cache_write = (
            int(raw_cache_write)
            if isinstance(raw_cache_write, (int, float)) and not isinstance(raw_cache_write, bool)
            else 0
        )

        reasoning_tokens = 0
        out_details = _get_field(raw_usage, "output_tokens_details")
        for candidate in (
            _get_field(raw_usage, "thinking_tokens"),
            _get_field(raw_usage, "reasoning_tokens"),
            _get_field(out_details, "reasoning_tokens"),
            _get_field(out_details, "thinking_tokens"),
        ):
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                reasoning_tokens = int(candidate)
                break

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
            reasoning_tokens=reasoning_tokens,
        )
