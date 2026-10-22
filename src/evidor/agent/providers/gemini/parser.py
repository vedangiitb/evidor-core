"""Response parser for Google Gemini."""

from typing import Any

from evidor.models import GenerationResponse, ToolCall
from ..base import build_token_usage, safe_parse_tool_arguments


class GeminiResponseParser:
    """Parses raw Google GenAI interaction responses into GenerationResponse objects."""

    def __init__(self, model: str) -> None:
        self.model = model

    def parse(self, response: Any) -> GenerationResponse:
        """Parse a Gemini interaction object into GenerationResponse."""
        resp_id = getattr(response, "id", None) or (response.get("id") if isinstance(response, dict) else None)
        resp_id = resp_id if isinstance(resp_id, str) else None
        raw_output_text = (
            getattr(response, "output_text", None)
            or (response.get("output_text") if isinstance(response, dict) else None)
        )
        output_text = raw_output_text if isinstance(raw_output_text, str) else ""

        steps = (
            getattr(response, "steps", None)
            or (response.get("steps") if isinstance(response, dict) else None)
            or []
        )
        text, tool_calls, thought_blocks, thought_parts = self.parse_steps(steps, output_text)

        if not text and hasattr(response, "text") and isinstance(response.text, str):
            text = response.text
        elif not text and isinstance(response, dict) and isinstance(response.get("text"), str):
            text = response["text"]

        thinking_text = "\n".join(thought_parts) if thought_parts else None
        usage_data = (
            getattr(response, "usage", None)
            or (response.get("usage") if isinstance(response, dict) else None)
        )
        usage = self.parse_usage(usage_data)

        return GenerationResponse(
            id=resp_id,
            text=text,
            model=self.model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(thought_blocks),
        )

    def parse_steps(
        self, steps: list[Any], initial_text: str
    ) -> tuple[str, list[ToolCall], list[dict[str, Any]], list[str]]:
        """Extract text, function calls, thought blocks, and thought summaries from response steps."""
        text_parts: list[str] = []
        if initial_text and isinstance(initial_text, str):
            text_parts.append(initial_text)

        tool_calls: list[ToolCall] = []
        thought_blocks: list[dict[str, Any]] = []
        thought_parts: list[str] = []

        for step in steps:
            step_type = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
            if step_type == "function_call":
                raw_call_id = getattr(step, "id", None) or (step.get("id") if isinstance(step, dict) else None)
                call_id = raw_call_id or f"call_{len(tool_calls)+1}"
                name = getattr(step, "name", None) or (step.get("name") if isinstance(step, dict) else "") or ""
                raw_args = (
                    getattr(step, "arguments", None)
                    or (step.get("arguments") if isinstance(step, dict) else {})
                    or {}
                )
                args = safe_parse_tool_arguments(raw_args)
                sig = getattr(step, "signature", None) or (step.get("signature") if isinstance(step, dict) else None)

                tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
                thought_blocks.append({
                    "type": "function_call",
                    "id": call_id,
                    "name": name,
                    "args": args,
                    "thought_signature": sig,
                    "_raw_part": step,
                })
            elif step_type == "thought":
                summary = getattr(step, "summary", None) or (step.get("summary") if isinstance(step, dict) else None)
                sig = getattr(step, "signature", None) or (step.get("signature") if isinstance(step, dict) else None)
                t_str = ""
                if isinstance(summary, list):
                    for s in summary:
                        val = getattr(s, "text", None) if not isinstance(s, dict) else s.get("text")
                        if val and isinstance(val, str):
                            t_str += val
                elif isinstance(summary, dict):
                    val = summary.get("text")
                    if val and isinstance(val, str):
                        t_str = val
                elif isinstance(summary, str):
                    t_str = summary
                elif getattr(step, "text", None) or (step.get("text") if isinstance(step, dict) else None):
                    val = getattr(step, "text", None) or (step.get("text") if isinstance(step, dict) else None)
                    if isinstance(val, str):
                        t_str = val

                if t_str:
                    thought_parts.append(t_str)
                thought_blocks.append({
                    "type": "thought",
                    "text": t_str,
                    "thought": True,
                    "thought_signature": sig,
                    "_raw_part": step,
                })
            elif step_type == "model_output":
                if not initial_text:
                    content = (
                        getattr(step, "content", None)
                        or (step.get("content") if isinstance(step, dict) else None)
                    )
                    if isinstance(content, list):
                        for c in content:
                            val = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                            if val and isinstance(val, str):
                                text_parts.append(val)
                    elif isinstance(content, dict):
                        val = content.get("text")
                        if val and isinstance(val, str):
                            text_parts.append(val)
                    elif isinstance(content, str) and content:
                        text_parts.append(content)

        text = "".join(text_parts)
        return text, tool_calls, thought_blocks, thought_parts

    def parse_usage(self, usage_obj: Any) -> Any:
        """Parse token usage statistics from Gemini interaction usage."""
        if usage_obj is None:
            return None

        def _get_field(*keys: str, default: Any = None) -> Any:
            if isinstance(usage_obj, dict):
                for k in keys:
                    if k in usage_obj and usage_obj[k] is not None:
                        return usage_obj[k]
            for k in keys:
                val = getattr(usage_obj, k, None)
                if val is not None:
                    return val
            return default

        pt = _get_field("total_input_tokens", "input_tokens", "prompt_tokens")
        ct = _get_field("total_output_tokens", "output_tokens", "completion_tokens")
        tt = _get_field("total_tokens")
        cached = _get_field("total_cached_tokens", "cached_tokens", default=0) or 0
        reasoning = _get_field("total_thought_tokens", "thought_tokens", "reasoning_tokens", default=0) or 0

        if not (isinstance(pt, int) or isinstance(ct, int) or isinstance(tt, int)):
            return None

        prompt_tokens = int(pt) if isinstance(pt, int) else 0
        completion_tokens = int(ct) if isinstance(ct, int) else 0

        return build_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(tt) if isinstance(tt, int) else (prompt_tokens + completion_tokens),
            cached_tokens=int(cached) if isinstance(cached, int) else 0,
            reasoning_tokens=int(reasoning) if isinstance(reasoning, int) else 0,
        )

