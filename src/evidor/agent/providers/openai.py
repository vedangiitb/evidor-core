import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, overload

from evidor.models import (
    AsyncStreamResponse,
    GenerationRequest,
    GenerationResponse,
    Message,
    StreamChunk,
    StreamResponse,
    ToolCall,
)
from evidor.utils import check_sync_allowed
from .base import (
    BaseProvider,
    ToolCallAccumulator,
    build_token_usage,
    is_client_closed,
    safe_parse_tool_arguments,
    sort_tools,
)


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
        base_url: str | None = None,
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
        self._base_url = base_url
        self._sync_client: Any = None
        self._default_async_client: Any = None

    @property
    def context_strategy(self) -> str:
        return "response_chain"

    @property
    def compact_threshold(self) -> int | None:
        return self._compact_threshold

    @property
    def base_url(self) -> str | None:
        """The custom base URL for OpenAI-compatible endpoints, if configured."""
        return self._base_url

    def with_model(self, model: str) -> "OpenAIProvider":
        return OpenAIProvider(
            model=model,
            api_key=self._api_key,
            thinking=self._thinking,
            compact_threshold=self._compact_threshold,
            base_url=self._base_url,
        )

    @overload
    def generate(self, request: GenerationRequest, *, stream: Literal[False] = False) -> GenerationResponse: ...

    @overload
    def generate(self, request: GenerationRequest, *, stream: Literal[True]) -> StreamResponse: ...

    @overload
    def generate(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse: ...

    def generate(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse:
        check_sync_allowed(
            "OpenAIProvider.generate() cannot be called from a running event loop. "
            "Use 'await provider.generate_async()' instead."
        )

        client = self._get_client()
        should_stream = stream or request.stream
        payload = self._build_request_payload(request)
        if should_stream:
            payload["stream"] = True
            raw_stream = client.responses.create(**payload)
            return StreamResponse(self._create_stream(raw_stream))

        response = client.responses.create(**payload)
        return self._parse_response(response)

    @overload
    async def generate_async(
        self, request: GenerationRequest, *, stream: Literal[False] = False
    ) -> GenerationResponse: ...

    @overload
    async def generate_async(
        self, request: GenerationRequest, *, stream: Literal[True]
    ) -> AsyncStreamResponse: ...

    @overload
    async def generate_async(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse: ...

    async def generate_async(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse:
        client = self._get_async_client()
        should_stream = stream or request.stream
        if client is not None:
            payload = self._build_request_payload(request)
            if should_stream:
                payload["stream"] = True
                raw_stream = await client.responses.create(**payload)
                return AsyncStreamResponse(self._create_async_stream(raw_stream))

            response = await client.responses.create(**payload)
            return self._parse_response(response)
        return await asyncio.to_thread(self.generate, request, stream=stream)

    def _create_stream(self, raw_stream: Any) -> Iterator[StreamChunk]:
        tool_acc = ToolCallAccumulator()
        for event in raw_stream:
            ev_type = getattr(event, "type", None)
            if ev_type in ("response.output_text.delta", "response.text.delta"):
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield StreamChunk(text=delta, raw_event=event)
            elif ev_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield StreamChunk(thinking=delta, raw_event=event)
            elif ev_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if item and getattr(item, "type", None) == "function_call":
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or "call_1"
                    name = getattr(item, "name", "") or ""
                    tool_acc.start_call(call_id, name)
            elif ev_type == "response.function_call_arguments.delta":
                delta = getattr(event, "delta", "") or ""
                call_id = getattr(event, "item_id", None) or getattr(event, "call_id", None) or "call_1"
                tool_acc.add_arguments_delta(call_id, delta)
            elif ev_type in ("response.function_call_arguments.done", "response.output_item.done"):
                item = getattr(event, "item", None)
                call_id = getattr(event, "item_id", None) or getattr(item, "call_id", None) if item else None
                if not call_id and item:
                    call_id = getattr(item, "id", None)
                if call_id:
                    tc = tool_acc.complete_call(call_id)
                    if tc:
                        yield StreamChunk(tool_calls=(tc,), raw_event=event)
            elif ev_type == "response.completed":
                final_resp = self._parse_response(getattr(event, "response", None))
                yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)

        remaining_tools = tool_acc.finalize_all()
        if remaining_tools:
            yield StreamChunk(tool_calls=tuple(remaining_tools))

    async def _create_async_stream(self, raw_stream: Any) -> AsyncIterator[StreamChunk]:
        tool_acc = ToolCallAccumulator()
        async for event in raw_stream:
            ev_type = getattr(event, "type", None)
            if ev_type in ("response.output_text.delta", "response.text.delta"):
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield StreamChunk(text=delta, raw_event=event)
            elif ev_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield StreamChunk(thinking=delta, raw_event=event)
            elif ev_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if item and getattr(item, "type", None) == "function_call":
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or "call_1"
                    name = getattr(item, "name", "") or ""
                    tool_acc.start_call(call_id, name)
            elif ev_type == "response.function_call_arguments.delta":
                delta = getattr(event, "delta", "") or ""
                call_id = getattr(event, "item_id", None) or getattr(event, "call_id", None) or "call_1"
                tool_acc.add_arguments_delta(call_id, delta)
            elif ev_type in ("response.function_call_arguments.done", "response.output_item.done"):
                item = getattr(event, "item", None)
                call_id = getattr(event, "item_id", None) or getattr(item, "call_id", None) if item else None
                if not call_id and item:
                    call_id = getattr(item, "id", None)
                if call_id:
                    tc = tool_acc.complete_call(call_id)
                    if tc:
                        yield StreamChunk(tool_calls=(tc,), raw_event=event)
            elif ev_type == "response.completed":
                final_resp = self._parse_response(getattr(event, "response", None))
                yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)

        remaining_tools = tool_acc.finalize_all()
        if remaining_tools:
            yield StreamChunk(tool_calls=tuple(remaining_tools))

    def close(self) -> None:
        with self._lock:
            client = self._sync_client
            self._sync_client = None

        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    async def close_async(self) -> None:
        self.close()
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        with self._lock:
            clients_to_close = list(self._async_clients.items())
            self._async_clients.clear()
            default_client = self._default_async_client
            self._default_async_client = None

        for loop, client in clients_to_close:
            try:
                if loop is current_loop:
                    await client.close()
                elif loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(client.close(), loop)
                    await asyncio.wrap_future(fut)
                else:
                    await client.close()
            except Exception:
                pass

        if default_client is not None:
            try:
                await default_client.close()
            except Exception:
                pass

    def _get_client(self) -> Any:
        with self._lock:
            if not is_client_closed(self._sync_client):
                return self._sync_client

            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError("Install OpenAI support with: pip install 'evidor[openai]'") from error

            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url is not None:
                client_kwargs["base_url"] = self._base_url
            client = OpenAI(**client_kwargs)
            if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
                raise RuntimeError(
                    "Modern OpenAI Responses API (client.responses.create) is required. "
                    "Please update openai to the latest version."
                )
            self._sync_client = client
            return client

    def _get_async_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except (ImportError, AttributeError):
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._lock:
            if not hasattr(self, "_async_clients"):
                self._async_clients: dict[asyncio.AbstractEventLoop, Any] = {}

            # Prune closed loops
            dead_loops = [l for l in self._async_clients if l.is_closed()]
            for l in dead_loops:
                del self._async_clients[l]

            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url is not None:
                client_kwargs["base_url"] = self._base_url

            if loop is not None:
                client = self._async_clients.get(loop)
                if is_client_closed(client):
                    client = AsyncOpenAI(**client_kwargs)
                    self._async_clients[loop] = client
            else:
                if is_client_closed(self._default_async_client):
                    self._default_async_client = AsyncOpenAI(**client_kwargs)
                client = self._default_async_client

        if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
            return None
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
