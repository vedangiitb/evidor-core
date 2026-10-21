"""Response parser for OpenAI Responses API."""

from typing import Any

from evidor.models import GenerationResponse, ToolCall
from ..base import build_token_usage, safe_parse_tool_arguments


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
                t = item.get("text") or item.get("content")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            else:
                t = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
    elif isinstance(val, dict):
        t = val.get("text") or val.get("content")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    else:
        t = getattr(val, "text", None) or getattr(val, "content", None)
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    return texts


class OpenAIResponseParser:
    """Parses raw OpenAI Responses API responses into GenerationResponse objects."""

    def __init__(self, model: str) -> None:
        self.model = model

    def parse(self, response: Any) -> GenerationResponse:
        """Parse an OpenAI response object into GenerationResponse."""
        resp_id = getattr(response, "id", None)
        resp_id = resp_id if isinstance(resp_id, str) else None
        text = getattr(response, "output_text", None) or ""

        tool_calls, thinking_parts, compaction_text = self.parse_output_items(
            getattr(response, "output", None) or []
        )

        raw_resp_thinking = getattr(response, "reasoning", None) or getattr(response, "thinking", None)
        if isinstance(raw_resp_thinking, str) and raw_resp_thinking:
            thinking_parts.append(raw_resp_thinking)

        thinking_text = "\n".join(thinking_parts) if thinking_parts else None
        usage = self.parse_usage(getattr(response, "usage", None))

        return GenerationResponse(
            id=resp_id,
            text=text,
            model=self.model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            compaction=compaction_text,
        )

    def parse_output_items(
        self, output_items: list[Any]
    ) -> tuple[list[ToolCall], list[str], str | None]:
        """Extract function calls, reasoning snippets, and compaction text from response items."""
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        compaction_part: str | None = None

        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type in ("reasoning", "thought"):
                for attr_name in ("summary", "content", "text"):
                    extracted = extract_reasoning_texts(getattr(item, attr_name, None))
                    if extracted:
                        thinking_parts.extend(extracted)
                        break
            elif item_type == "function_call":
                args = safe_parse_tool_arguments(item.arguments)
                raw_call_id = getattr(item, "call_id", None)
                if not isinstance(raw_call_id, str):
                    raw_call_id = getattr(item, "id", None)
                call_id = raw_call_id if isinstance(raw_call_id, str) else f"call_{len(tool_calls)+1}"
                tool_calls.append(ToolCall(id=call_id, name=item.name, arguments=args))
            elif item_type == "compaction":
                c = getattr(item, "content", None) or getattr(item, "encrypted_content", None) or getattr(item, "text", None)
                if c:
                    compaction_part = str(c)

        return tool_calls, thinking_parts, compaction_part

    def parse_usage(self, raw_usage: Any) -> Any:
        """Parse token usage statistics from OpenAI Responses API usage."""
        if raw_usage is None:
            return None

        pt = getattr(raw_usage, "prompt_tokens", None)
        if pt is None:
            pt = getattr(raw_usage, "input_tokens", None)
        ct = getattr(raw_usage, "completion_tokens", None)
        if ct is None:
            ct = getattr(raw_usage, "output_tokens", None)
        tt = getattr(raw_usage, "total_tokens", None)

        if not (isinstance(pt, int) or isinstance(ct, int) or isinstance(tt, int)):
            return None

        prompt_tokens = int(pt) if isinstance(pt, int) else 0
        completion_tokens = int(ct) if isinstance(ct, int) else 0

        cached_tokens = 0
        pt_details = getattr(raw_usage, "prompt_tokens_details", None) or getattr(raw_usage, "input_tokens_details", None)
        if pt_details is not None:
            cached_tokens = getattr(pt_details, "cached_tokens", 0) or 0
            if not cached_tokens:
                cached_tokens = getattr(pt_details, "prompt_cache_hit_tokens", 0) or 0
        if not cached_tokens:
            cached_tokens = getattr(raw_usage, "prompt_cache_hit_tokens", 0) or 0

        reasoning_tokens = 0
        ct_details = getattr(raw_usage, "completion_tokens_details", None) or getattr(raw_usage, "output_tokens_details", None)
        if ct_details is not None:
            reasoning_tokens = getattr(ct_details, "reasoning_tokens", 0) or 0

        return build_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(tt) if isinstance(tt, int) else None,
            cached_tokens=int(cached_tokens) if isinstance(cached_tokens, int) else 0,
            reasoning_tokens=int(reasoning_tokens) if isinstance(reasoning_tokens, int) else 0,
        )

