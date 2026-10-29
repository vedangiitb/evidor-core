"""Response parser for OpenAI Responses API."""

from typing import Any, NamedTuple

from evidor.models import GenerationResponse, ToolCall
from ..base import build_token_usage, safe_parse_tool_arguments


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Retrieve an attribute or dict key safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_reasoning_texts(val: Any) -> list[str]:
    """Extract string snippets from diverse reasoning or thought payload shapes."""
    if not val:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    texts: list[str] = []
    if isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    texts.append(s)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or item.get("summary")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            else:
                t = (
                    getattr(item, "text", None)
                    or getattr(item, "content", None)
                    or getattr(item, "summary", None)
                )
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
    elif isinstance(val, dict):
        t = val.get("text") or val.get("content") or val.get("summary")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    else:
        t = (
            getattr(val, "text", None)
            or getattr(val, "content", None)
            or getattr(val, "summary", None)
        )
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    return texts


class ParsedOutputItems(NamedTuple):
    """Structured extraction results from response output items."""

    tool_calls: list[ToolCall]
    thinking_parts: list[str]
    compaction: str | None = None
    text_parts: list[str] = []
    refusal_parts: list[str] = []
    thought_blocks: list[Any] = []


class OpenAIResponseParser:
    """Parses raw OpenAI Responses API responses into GenerationResponse objects."""

    def __init__(self, model: str) -> None:
        self.model = model

    def parse(self, response: Any) -> GenerationResponse:
        """Parse an OpenAI response object into GenerationResponse."""
        raw_id = _get_field(response, "id")
        resp_id = raw_id if isinstance(raw_id, str) else None

        raw_output = _get_field(response, "output")
        output_items = raw_output if isinstance(raw_output, (list, tuple)) else []
        parsed = self.parse_output_items(output_items)

        output_text = _get_field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            text = output_text
        elif parsed.text_parts:
            text = "".join(parsed.text_parts)
        elif parsed.refusal_parts:
            text = "\n".join(parsed.refusal_parts)
        else:
            text = ""

        thinking_parts = list(parsed.thinking_parts)
        seen_thinking = set(thinking_parts)

        raw_resp_thinking = _get_field(response, "reasoning") or _get_field(response, "thinking")
        if isinstance(raw_resp_thinking, str) and raw_resp_thinking.strip():
            s = raw_resp_thinking.strip()
            if s not in seen_thinking:
                seen_thinking.add(s)
                thinking_parts.append(s)
        elif raw_resp_thinking is not None:
            summary_val = _get_field(raw_resp_thinking, "summary")
            if isinstance(summary_val, (str, list, tuple, dict)):
                for snippet in extract_reasoning_texts(summary_val):
                    if snippet not in seen_thinking:
                        seen_thinking.add(snippet)
                        thinking_parts.append(snippet)

        thinking_text = "\n".join(thinking_parts) if thinking_parts else None
        usage = self.parse_usage(_get_field(response, "usage"))

        return GenerationResponse(
            id=resp_id,
            text=text,
            model=self.model,
            tool_calls=tuple(parsed.tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(parsed.thought_blocks),
            compaction=parsed.compaction,
        )

    def parse_output_items(self, output_items: list[Any]) -> ParsedOutputItems:
        """Extract function calls, reasoning snippets, compaction, text, and refusal from output items."""
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        seen_thinking: set[str] = set()
        compaction_part: str | None = None
        text_parts: list[str] = []
        refusal_parts: list[str] = []
        thought_blocks: list[Any] = []

        if not output_items or not isinstance(output_items, (list, tuple)):
            return ParsedOutputItems(
                tool_calls=tool_calls,
                thinking_parts=thinking_parts,
                compaction=compaction_part,
                text_parts=text_parts,
                refusal_parts=refusal_parts,
                thought_blocks=thought_blocks,
            )

        for item in output_items:
            raw_type = _get_field(item, "type")
            item_type = raw_type if isinstance(raw_type, str) else None
            if item_type in ("reasoning", "thought"):
                thought_blocks.append(item)
                for attr_name in ("summary", "content", "text"):
                    val = _get_field(item, attr_name)
                    if isinstance(val, (str, list, tuple, dict)):
                        for snippet in extract_reasoning_texts(val):
                            if snippet not in seen_thinking:
                                seen_thinking.add(snippet)
                                thinking_parts.append(snippet)
            elif item_type in ("function_call", "custom_tool_call"):
                raw_args = _get_field(item, "arguments")
                args = (
                    safe_parse_tool_arguments(raw_args)
                    if isinstance(raw_args, (str, dict)) or (hasattr(raw_args, "items") and not isinstance(raw_args, type))
                    else {}
                )
                raw_call_id = _get_field(item, "call_id")
                if not isinstance(raw_call_id, str) or not raw_call_id.strip():
                    raw_call_id = _get_field(item, "id")
                call_id = (
                    raw_call_id.strip()
                    if isinstance(raw_call_id, str) and raw_call_id.strip()
                    else f"call_{len(tool_calls)+1}"
                )
                raw_name = _get_field(item, "name")
                name = raw_name if isinstance(raw_name, str) and raw_name.strip() else "tool"
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
            elif item_type == "compaction":
                c = None
                for attr_name in ("encrypted_content", "content", "text"):
                    val = _get_field(item, attr_name)
                    if isinstance(val, str) and val.strip():
                        c = val.strip()
                        break
                if c is None:
                    for attr_name in ("encrypted_content", "content", "text"):
                        val = _get_field(item, attr_name)
                        if val is not None and not isinstance(val, (type, bool)) and type(val).__name__ != "MagicMock":
                            c = str(val)
                            break
                if c:
                    compaction_part = c
            elif item_type == "message":
                contents = _get_field(item, "content")
                if isinstance(contents, str) and contents.strip():
                    text_parts.append(contents)
                elif isinstance(contents, (list, tuple)):
                    for part in contents:
                        if isinstance(part, str) and part.strip():
                            text_parts.append(part)
                        elif part is not None:
                            raw_part_type = _get_field(part, "type")
                            part_type = raw_part_type if isinstance(raw_part_type, str) else None
                            if part_type == "output_text":
                                t = _get_field(part, "text")
                                if isinstance(t, str) and t:
                                    text_parts.append(t)
                            elif part_type == "refusal":
                                r = _get_field(part, "refusal")
                                if not isinstance(r, str):
                                    r = _get_field(part, "text")
                                if isinstance(r, str) and r:
                                    refusal_parts.append(r)
                            else:
                                t = _get_field(part, "text")
                                if isinstance(t, str) and t:
                                    text_parts.append(t)

        return ParsedOutputItems(
            tool_calls=tool_calls,
            thinking_parts=thinking_parts,
            compaction=compaction_part,
            text_parts=text_parts,
            refusal_parts=refusal_parts,
            thought_blocks=thought_blocks,
        )

    def parse_usage(self, raw_usage: Any) -> Any:
        """Parse token usage statistics from OpenAI Responses API usage."""
        if raw_usage is None:
            return None

        pt = _get_field(raw_usage, "prompt_tokens")
        if not isinstance(pt, int) or isinstance(pt, bool):
            pt = _get_field(raw_usage, "input_tokens")
        ct = _get_field(raw_usage, "completion_tokens")
        if not isinstance(ct, int) or isinstance(ct, bool):
            ct = _get_field(raw_usage, "output_tokens")
        tt = _get_field(raw_usage, "total_tokens")

        has_pt = isinstance(pt, int) and not isinstance(pt, bool)
        has_ct = isinstance(ct, int) and not isinstance(ct, bool)
        has_tt = isinstance(tt, int) and not isinstance(tt, bool)

        if not (has_pt or has_ct or has_tt):
            return None

        prompt_tokens = int(pt) if has_pt else 0
        completion_tokens = int(ct) if has_ct else 0

        cached_tokens = 0
        cache_creation_tokens = 0
        pt_details = _get_field(raw_usage, "prompt_tokens_details") or _get_field(raw_usage, "input_tokens_details")
        if pt_details is not None:
            raw_cached = _get_field(pt_details, "cached_tokens", 0)
            if isinstance(raw_cached, int) and not isinstance(raw_cached, bool):
                cached_tokens = int(raw_cached)
            if not cached_tokens:
                raw_hit = _get_field(pt_details, "prompt_cache_hit_tokens", 0)
                if isinstance(raw_hit, int) and not isinstance(raw_hit, bool):
                    cached_tokens = int(raw_hit)

            raw_write = _get_field(pt_details, "cache_write_tokens", 0)
            if not isinstance(raw_write, int) or isinstance(raw_write, bool) or raw_write == 0:
                raw_write = _get_field(pt_details, "cache_creation_tokens", 0)
            if isinstance(raw_write, int) and not isinstance(raw_write, bool):
                cache_creation_tokens = int(raw_write)

        if not cached_tokens:
            raw_hit = _get_field(raw_usage, "prompt_cache_hit_tokens", 0)
            if isinstance(raw_hit, int) and not isinstance(raw_hit, bool):
                cached_tokens = int(raw_hit)

        reasoning_tokens = 0
        ct_details = _get_field(raw_usage, "completion_tokens_details") or _get_field(raw_usage, "output_tokens_details")
        if ct_details is not None:
            raw_reasoning = _get_field(ct_details, "reasoning_tokens", 0)
            if isinstance(raw_reasoning, int) and not isinstance(raw_reasoning, bool):
                reasoning_tokens = int(raw_reasoning)

        return build_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(tt) if has_tt else None,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
