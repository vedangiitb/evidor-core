"""Anthropic adapter."""

from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, ToolCall


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        enable_prompt_caching: bool = True,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._enable_prompt_caching = enable_prompt_caching

    @property
    def model(self) -> str:
        return self._model

    @property
    def enable_prompt_caching(self) -> bool:
        return self._enable_prompt_caching

    def with_model(self, model: str) -> "AnthropicProvider":
        return AnthropicProvider(
            model=model,
            api_key=self._api_key,
            max_tokens=self._max_tokens,
            enable_prompt_caching=self._enable_prompt_caching,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from anthropic import Anthropic
        except ImportError as error:
            raise ImportError("Install Anthropic support with: pip install 'evidor[anthropic]'") from error

        client = Anthropic(api_key=self._api_key, max_retries=0)
        system_prompt = "\n\n".join(message.content for message in request.messages if message.role == "system")

        formatted_messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                })
            elif message.role == "assistant" and message.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if message.content:
                    content_blocks.append({"type": "text", "text": message.content})
                for tc in message.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                formatted_messages.append({"role": "assistant", "content": content_blocks})
            else:
                formatted_messages.append({"role": message.role, "content": message.content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt or None,
            "messages": formatted_messages,
        }

        if self._enable_prompt_caching:
            kwargs["cache_control"] = {"type": "ephemeral"}

        if request.tools:
            sorted_tools = sorted(request.tools, key=lambda t: t.name)
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in sorted_tools
            ]

        response_msg = client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in getattr(response_msg, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", f"call_{len(tool_calls)+1}"),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}),
                    )
                )

        usage = None
        raw_usage = getattr(response_msg, "usage", None)
        if raw_usage is not None:
            try:
                from evidor.telemetry.events import TokenUsage

                it = getattr(raw_usage, "input_tokens", None)
                ot = getattr(raw_usage, "output_tokens", None)
                cache_read = getattr(raw_usage, "cache_read_input_tokens", 0) or 0
                cache_write = getattr(raw_usage, "cache_creation_input_tokens", 0) or 0

                if isinstance(it, int) or isinstance(ot, int):
                    prompt_tokens = (int(it) if isinstance(it, int) else 0) + cache_read
                    completion_tokens = int(ot) if isinstance(ot, int) else 0
                    total_tokens = prompt_tokens + completion_tokens
                    usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=cache_read,
                        cache_creation_tokens=cache_write,
                    )
            except Exception:
                pass

        return GenerationResponse(
            text="".join(text_parts),
            model=self._model,
            tool_calls=tuple(tool_calls),
            usage=usage,
        )
