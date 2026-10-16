"""OpenAI adapter."""

import json
from collections.abc import Sequence
from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, Message, ToolCall
from .base import BaseProvider, build_token_usage, safe_parse_tool_arguments, sort_tools


def _extract_reasoning_texts(val: Any) -> list[str]:
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


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI using the modern Responses API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking: Any = None,
        compact_threshold: int | None = None,
    ) -> None:
        if thinking is not None:
            from evidor.models import OpenAIThinkingConfig

            thinking_cfg: OpenAIThinkingConfig | None = (
                thinking
                if isinstance(thinking, OpenAIThinkingConfig)
                else OpenAIThinkingConfig.from_value(thinking)
            )
            thinking_cfg.validate()
        else:
            thinking_cfg = None

        super().__init__(model=model, api_key=api_key, thinking=thinking_cfg)
        self._compact_threshold = compact_threshold

    @property
    def context_strategy(self) -> str:
        return "response_chain"

    @property
    def compact_threshold(self) -> int | None:
        return self._compact_threshold

    def with_model(self, model: str) -> "OpenAIProvider":
        return OpenAIProvider(
            model=model,
            api_key=self._api_key,
            thinking=self._thinking,
            compact_threshold=self._compact_threshold,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        payload = self._build_request_payload(request)
        response = client.responses.create(**payload)
        return self._parse_response(response)

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("Install OpenAI support with: pip install 'evidor[openai]'") from error

        client = OpenAI(api_key=self._api_key)
        if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
            raise RuntimeError(
                "Modern OpenAI Responses API (client.responses.create) is required. "
                "Please update openai to the latest version."
            )
        return client

    def _build_request_payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "input": self._build_input_items(request.messages),
        }

        if request.system:
            payload["instructions"] = request.system
        if request.previous_response_id:
            payload["previous_response_id"] = request.previous_response_id
        if request.conversation_id:
            payload["conversation"] = request.conversation_id
        if request.store is not None:
            payload["store"] = request.store

        if self._compact_threshold is not None:
            payload["context_management"] = [
                {"type": "compaction", "compact_threshold": self._compact_threshold}
            ]

        if self._thinking and self._thinking.enabled:
            reasoning_config: dict[str, Any] = {}
            if self._thinking.effort:
                reasoning_config["effort"] = self._thinking.effort
            summary_val = getattr(self._thinking, "summary", None)
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

    def _build_input_items(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
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

    def _parse_response(self, response: Any) -> GenerationResponse:
        resp_id = getattr(response, "id", None)
        resp_id = resp_id if isinstance(resp_id, str) else None
        text = getattr(response, "output_text", None) or ""

        tool_calls, thinking_parts, compaction_text = self._parse_output_items(getattr(response, "output", None) or [])

        raw_resp_thinking = getattr(response, "reasoning", None) or getattr(response, "thinking", None)
        if isinstance(raw_resp_thinking, str) and raw_resp_thinking:
            thinking_parts.append(raw_resp_thinking)

        thinking_text = "\n".join(thinking_parts) if thinking_parts else None
        usage = self._parse_usage(getattr(response, "usage", None))

        return GenerationResponse(
            id=resp_id,
            text=text,
            model=self._model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            compaction=compaction_text,
        )

    def _parse_output_items(
        self, output_items: list[Any]
    ) -> tuple[list[ToolCall], list[str], str | None]:
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        compaction_part: str | None = None

        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type in ("reasoning", "thought"):
                for attr_name in ("summary", "content", "text"):
                    extracted = _extract_reasoning_texts(getattr(item, attr_name, None))
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

    def _parse_usage(self, raw_usage: Any) -> Any:
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
