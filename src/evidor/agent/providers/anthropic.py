"""Anthropic adapter."""

from collections.abc import Sequence
from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, Message, ToolCall
from .base import BaseProvider, build_token_usage, extract_system_prompt, sort_tools


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic Claude using the Messages API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
        enable_prompt_caching: bool = True,
        thinking: Any = None,
    ) -> None:
        if thinking is not None:
            from evidor.models import AnthropicThinkingConfig

            thinking_cfg: AnthropicThinkingConfig | None = (
                thinking
                if isinstance(thinking, AnthropicThinkingConfig)
                else AnthropicThinkingConfig.from_value(thinking)
            )
            thinking_cfg.validate()
        else:
            thinking_cfg = None

        super().__init__(model=model, api_key=api_key, thinking=thinking_cfg)
        self._enable_prompt_caching = enable_prompt_caching

        if max_tokens is None:
            if self._thinking and self._thinking.enabled and self._thinking.budget_tokens:
                max_tokens = self._thinking.budget_tokens + 2048
            else:
                max_tokens = 1024
        self._max_tokens = max_tokens

    @property
    def context_strategy(self) -> str:
        return "prompt_caching"

    @property
    def enable_prompt_caching(self) -> bool:
        return self._enable_prompt_caching

    def with_model(self, model: str) -> "AnthropicProvider":
        return AnthropicProvider(
            model=model,
            api_key=self._api_key,
            max_tokens=self._max_tokens,
            enable_prompt_caching=self._enable_prompt_caching,
            thinking=self._thinking,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        payload = self._build_request_payload(request)
        response_msg = client.messages.create(**payload)
        return self._parse_response(response_msg)

    def _get_client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as error:
            raise ImportError("Install Anthropic support with: pip install 'evidor[anthropic]'") from error

        return Anthropic(api_key=self._api_key)

    def _build_request_payload(self, request: GenerationRequest) -> dict[str, Any]:
        system_prompt = extract_system_prompt(request)
        formatted_messages = self._format_messages(request.messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt or None,
            "messages": formatted_messages,
        }

        if self._thinking and self._thinking.enabled:
            if self._thinking.type == "adaptive":
                payload["thinking"] = {"type": "adaptive"}
                if self._thinking.effort:
                    payload["output_config"] = {"effort": self._thinking.effort}
            elif self._thinking.budget_tokens:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking.budget_tokens,
                }

        if self._enable_prompt_caching:
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

    def _format_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
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
                if message.thought_blocks or message.tool_calls:
                    content_blocks: list[dict[str, Any]] = []
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

    def _parse_response(self, response_msg: Any) -> GenerationResponse:
        msg_id = getattr(response_msg, "id", None)
        msg_id = msg_id if isinstance(msg_id, str) else None

        text, tool_calls, thought_blocks, thought_parts = self._parse_content_blocks(
            getattr(response_msg, "content", [])
        )

        thinking_text = "".join(thought_parts) if thought_parts else None
        usage = self._parse_usage(getattr(response_msg, "usage", None))

        return GenerationResponse(
            id=msg_id,
            text=text,
            model=self._model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(thought_blocks),
        )

    def _parse_content_blocks(
        self, content_blocks: list[Any]
    ) -> tuple[str, list[ToolCall], list[dict[str, Any]], list[str]]:
        text_parts: list[str] = []
        thought_parts: list[str] = []
        thought_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
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

        return "".join(text_parts), tool_calls, thought_blocks, thought_parts

    def _parse_usage(self, raw_usage: Any) -> Any:
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
