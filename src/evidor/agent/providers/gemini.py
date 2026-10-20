import asyncio
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
    extract_system_prompt,
    safe_parse_tool_arguments,
    sort_tools,
)

_ALLOWED_GEMINI_SCHEMA_KEYS = {
    "type",
    "format",
    "title",
    "description",
    "default",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "any_of",
    "anyOf",
    "maximum",
    "minimum",
    "max_length",
    "maxLength",
    "min_length",
    "minLength",
    "pattern",
    "max_items",
    "maxItems",
    "min_items",
    "minItems",
    "max_properties",
    "maxProperties",
    "min_properties",
    "minProperties",
    "example",
    "property_ordering",
    "propertyOrdering",
}


def _sanitize_gemini_schema(schema: Any, types_mod: Any) -> Any:
    """Sanitize a JSON schema dictionary for Google GenAI types.Schema."""
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_sanitize_gemini_schema(item, types_mod) for item in schema]
        return schema

    cleaned: dict[str, Any] = {}

    # Handle 'type' if given as a list (union types, e.g. ['string', 'number', 'boolean'])
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        types_list = [t for t in raw_type if t != "null"]
        has_null = "null" in raw_type
        if len(types_list) == 1:
            cleaned["type"] = types_list[0]
            if has_null:
                cleaned["nullable"] = True
        elif len(types_list) > 1:
            subschemas = [{"type": t} for t in types_list]
            if has_null:
                cleaned["nullable"] = True
            cleaned["anyOf"] = subschemas
        elif has_null:
            cleaned["type"] = "NULL"
    elif raw_type is not None:
        cleaned["type"] = raw_type

    # Handle const -> enum
    if "const" in schema and "enum" not in schema:
        cleaned["enum"] = [schema["const"]]

    for k, v in schema.items():
        if k == "type":
            continue
        if k in ("oneOf", "one_of") and "anyOf" not in cleaned and "any_of" not in cleaned:
            cleaned["anyOf"] = (
                [_sanitize_gemini_schema(it, types_mod) for it in v]
                if isinstance(v, list)
                else _sanitize_gemini_schema(v, types_mod)
            )
            continue
        if k == "examples" and "example" not in schema and isinstance(v, list) and v:
            cleaned["example"] = v[0]
            continue

        if k == "properties" and isinstance(v, dict):
            cleaned["properties"] = {
                prop_name: _sanitize_gemini_schema(prop_s, types_mod)
                for prop_name, prop_s in v.items()
            }
        elif k == "items":
            if isinstance(v, list):
                if len(v) == 1:
                    cleaned["items"] = _sanitize_gemini_schema(v[0], types_mod)
                elif len(v) > 1:
                    cleaned["items"] = {"anyOf": [_sanitize_gemini_schema(it, types_mod) for it in v]}
            else:
                cleaned["items"] = _sanitize_gemini_schema(v, types_mod)
        elif k in _ALLOWED_GEMINI_SCHEMA_KEYS:
            cleaned[k] = _sanitize_gemini_schema(v, types_mod)

    return cleaned


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini using the Interactions API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cached_content: str | None = None,
        thinking: Any = None,
    ) -> None:
        if thinking is not None:
            from evidor.models import GeminiThinkingConfig

            thinking_cfg: GeminiThinkingConfig | None = (
                thinking
                if isinstance(thinking, GeminiThinkingConfig)
                else GeminiThinkingConfig.from_value(thinking)
            )
            thinking_cfg.validate()
        else:
            thinking_cfg = None

        super().__init__(model=model, api_key=api_key, thinking=thinking_cfg)
        self._cached_content = cached_content
        self._client: Any = None
        self._types: Any = None

    @property
    def context_strategy(self) -> str:
        return "interaction_chain"

    @property
    def cached_content(self) -> str | None:
        return self._cached_content

    def with_model(self, model: str) -> "GeminiProvider":
        return GeminiProvider(
            model=model,
            api_key=self._api_key,
            cached_content=self._cached_content,
            thinking=self._thinking,
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
            "GeminiProvider.generate() cannot be called from a running event loop. "
            "Use 'await provider.generate_async()' instead."
        )

        client, types = self._get_client_and_types()
        should_stream = stream or request.stream
        payload = self._build_request_payload(request, types)
        if should_stream:
            payload["stream"] = True
            raw_stream = client.interactions.create(**payload)
            return StreamResponse(self._create_stream(raw_stream))

        response = client.interactions.create(**payload)
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
        client, types = self._get_client_and_types()
        should_stream = stream or request.stream
        payload = self._build_request_payload(request, types)
        if hasattr(client, "aio") and hasattr(client.aio, "interactions") and hasattr(client.aio.interactions, "create"):
            if should_stream:
                payload["stream"] = True
                raw_stream = await client.aio.interactions.create(**payload)
                return AsyncStreamResponse(self._create_async_stream(raw_stream))

            response = await client.aio.interactions.create(**payload)
            return self._parse_response(response)
        return await asyncio.to_thread(self.generate, request, stream=stream)

    def _create_stream(self, raw_stream: Any) -> Iterator[StreamChunk]:
        tool_acc = ToolCallAccumulator()
        step_id_map: dict[str, str] = {}
        thought_blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for event in raw_stream:
            yield from self._process_gemini_event(event, tool_acc, step_id_map, thought_blocks, text_parts)

        remaining_in_flight = [
            tc for call_id in list(tool_acc._calls.keys())
            if (tc := tool_acc.complete_call(call_id)) is not None
        ]
        if remaining_in_flight:
            yield StreamChunk(tool_calls=tuple(remaining_in_flight))

    async def _create_async_stream(self, raw_stream: Any) -> AsyncIterator[StreamChunk]:
        tool_acc = ToolCallAccumulator()
        step_id_map: dict[str, str] = {}
        thought_blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        async for event in raw_stream:
            for chunk in self._process_gemini_event(event, tool_acc, step_id_map, thought_blocks, text_parts):
                yield chunk

        remaining_in_flight = [
            tc for call_id in list(tool_acc._calls.keys())
            if (tc := tool_acc.complete_call(call_id)) is not None
        ]
        if remaining_in_flight:
            yield StreamChunk(tool_calls=tuple(remaining_in_flight))

    def _process_gemini_event(
        self,
        event: Any,
        tool_acc: ToolCallAccumulator,
        step_id_map: dict[str, str],
        thought_blocks: list[dict[str, Any]],
        text_parts: list[str],
    ) -> Iterator[StreamChunk]:
        ev_type = getattr(event, "event_type", None) or getattr(event, "type", None)
        idx = str(getattr(event, "index", 0))

        delta_obj = getattr(event, "delta", None)
        if delta_obj is not None and ev_type not in (
            "step.start",
            "step_start",
            "step.stop",
            "step_stop",
            "interaction.completed",
            "interaction_completed",
        ):
            delta_type = getattr(delta_obj, "type", None)
            if delta_type == "text" or (hasattr(delta_obj, "text") and not hasattr(delta_obj, "arguments")):
                txt = getattr(delta_obj, "text", "") or ""
                if isinstance(txt, str) and txt:
                    text_parts.append(txt)
                    yield StreamChunk(text=txt, raw_event=event)
            elif delta_type in ("thought_summary", "thought"):
                th = getattr(delta_obj, "text", "") or getattr(delta_obj, "summary", "") or ""
                if isinstance(th, str) and th:
                    yield StreamChunk(thinking=th, raw_event=event)
            elif delta_type == "arguments_delta" or hasattr(delta_obj, "arguments"):
                call_id = (
                    getattr(event, "event_id", None)
                    or getattr(event, "step_id", None)
                    or step_id_map.get(idx)
                    or "call_1"
                )
                args_delta = getattr(delta_obj, "arguments", "") or getattr(delta_obj, "text", "") or ""
                if isinstance(args_delta, str) and args_delta:
                    tool_acc.add_arguments_delta(call_id, args_delta)
            elif delta_type in ("thought_signature", "thought"):
                sig = getattr(delta_obj, "signature", None)
                if isinstance(sig, str) and sig:
                    thought_blocks.append({
                        "type": "thought",
                        "thought": True,
                        "thought_signature": sig,
                    })

        if ev_type in ("step.start", "step_start"):
            step = getattr(event, "step", None)
            if step and getattr(step, "type", None) == "function_call":
                call_id = getattr(step, "id", None) or getattr(event, "event_id", None) or f"call_{idx}"
                name = getattr(step, "name", "") or ""
                step_id_map[idx] = call_id
                tool_acc.start_call(call_id, name)
                sig = getattr(step, "signature", None)
                if sig:
                    thought_blocks.append({
                        "type": "function_call",
                        "id": call_id,
                        "name": name,
                        "thought_signature": sig,
                    })
            elif step and getattr(step, "type", None) == "thought":
                step_id_map[idx] = f"thought_{idx}"
        elif ev_type in ("step.stop", "step_stop"):
            call_id = (
                getattr(event, "event_id", None)
                or getattr(event, "step_id", None)
                or step_id_map.pop(idx, None)
            )
            if call_id:
                tc = tool_acc.complete_call(call_id)
                if tc:
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)

        if ev_type in ("interaction.completed", "interaction_completed"):
            interaction = getattr(event, "interaction", event)
            final_resp = self._parse_response(interaction)
            accumulated_text = "".join(s for s in text_parts if isinstance(s, str))
            final_text = final_resp.text or accumulated_text
            final_tools = final_resp.tool_calls or tuple(tool_acc._completed)
            final_resp = GenerationResponse(
                id=final_resp.id,
                text=final_text,
                model=final_resp.model,
                tool_calls=final_tools,
                usage=final_resp.usage,
                thinking=final_resp.thinking,
                thought_blocks=tuple(thought_blocks) if thought_blocks else final_resp.thought_blocks,
                compaction=final_resp.compaction,
            )
            yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)
        elif hasattr(event, "text") and isinstance(event.text, str) and delta_obj is None:
            if event.text:
                text_parts.append(event.text)
                yield StreamChunk(text=event.text, raw_event=event)

    def close(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._types = None

        if client is not None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    async def close_async(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._types = None

        if client is not None:
            aio_client = getattr(client, "aio", None)
            close_async_fn = getattr(aio_client, "close", None)
            if callable(close_async_fn):
                try:
                    await close_async_fn()
                except Exception:
                    pass
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def _get_client_and_types(self) -> tuple[Any, Any]:
        with self._lock:
            if getattr(self, "_client", None) is not None and getattr(self, "_types", None) is not None:
                return self._client, self._types

            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise ImportError("Install Gemini support with: pip install 'evidor[gemini]'") from error

            client = genai.Client(api_key=self._api_key)
            if not hasattr(client, "interactions") or not hasattr(client.interactions, "create"):
                raise RuntimeError(
                    "Modern Google GenAI Interactions API (client.interactions.create) is required. "
                    "Please update google-genai to the latest version."
                )
            self._client = client
            self._types = types
            return self._client, self._types

    def _build_request_payload(self, request: GenerationRequest, types: Any) -> dict[str, Any]:
        system_prompt = extract_system_prompt(request)
        input_steps = self._format_input_steps(request.messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "input": input_steps,
        }

        if request.system:
            payload["system_instruction"] = request.system
        elif system_prompt:
            payload["system_instruction"] = system_prompt

        if request.previous_response_id:
            payload["previous_interaction_id"] = request.previous_response_id
        if request.store is not None:
            payload["store"] = request.store
        if self._cached_content:
            payload["cached_content"] = self._cached_content

        if request.tools:
            sorted_tools = sort_tools(request.tools)
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": _sanitize_gemini_schema(t.parameters, types) if t.parameters else None,
                }
                for t in sorted_tools
            ]

        if self._thinking is not None:
            gen_config: dict[str, Any] = {}
            if self._thinking.enabled:
                thinking_level = getattr(self._thinking, "thinking_level", None)
                if thinking_level:
                    gen_config["thinking_level"] = thinking_level
                elif self._thinking.thinking_budget is not None:
                    gen_config["thinking_level"] = "medium"
                if getattr(self._thinking, "include_thoughts", True):
                    gen_config["thinking_summaries"] = "auto"
            else:
                gen_config["thinking_level"] = "minimal"
            if gen_config:
                payload["generation_config"] = gen_config

        return payload

    def _format_input_steps(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        input_steps: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                input_steps.append({
                    "type": "function_result",
                    "call_id": message.tool_call_id or "call_1",
                    "name": message.name or "tool",
                    "result": {"result": message.content},
                })
            elif message.role == "assistant":
                if message.thought_blocks:
                    for tb in message.thought_blocks:
                        if isinstance(tb, dict):
                            tb_type = tb.get("type")
                            sig = tb.get("thought_signature") or tb.get("thoughtSignature") or tb.get("signature")
                            if tb_type == "thought" or tb.get("thought"):
                                input_steps.append({
                                    "type": "thought",
                                    "signature": sig,
                                    "summary": [{"type": "text", "text": tb.get("text", "")}],
                                })
                            elif tb_type == "function_call":
                                fc_step: dict[str, Any] = {
                                    "type": "function_call",
                                    "id": tb.get("id") or tb.get("call_id") or "call_1",
                                    "name": tb.get("name", ""),
                                    "arguments": tb.get("args") or tb.get("arguments") or {},
                                }
                                if sig:
                                    fc_step["signature"] = sig
                                input_steps.append(fc_step)
                        elif hasattr(tb, "type"):
                            input_steps.append(tb)

                if message.tool_calls:
                    existing_call_ids = {
                        step.get("id") for step in input_steps if isinstance(step, dict) and step.get("type") == "function_call"
                    }
                    for tc in message.tool_calls:
                        if tc.id not in existing_call_ids:
                            input_steps.append({
                                "type": "function_call",
                                "id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            })

                if message.content:
                    input_steps.append({
                        "type": "model_output",
                        "content": [{"type": "text", "text": message.content}],
                    })
            else:
                input_steps.append({
                    "type": "user_input",
                    "content": [{"type": "text", "text": message.content}],
                })
        return input_steps

    def _parse_response(self, response: Any) -> GenerationResponse:
        resp_id = getattr(response, "id", None)
        resp_id = resp_id if isinstance(resp_id, str) else None
        output_text = getattr(response, "output_text", None) or ""

        text, tool_calls, thought_blocks, thought_parts = self._parse_steps(
            getattr(response, "steps", None) or [], output_text
        )

        if not text and hasattr(response, "text") and isinstance(response.text, str):
            text = response.text
        if not text and not tool_calls and thought_parts:
            text = "".join(thought_parts)

        thinking_text = "\n".join(thought_parts) if thought_parts else None
        usage = self._parse_usage(getattr(response, "usage", None))

        return GenerationResponse(
            id=resp_id,
            text=text,
            model=self._model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(thought_blocks),
        )

    def _parse_steps(
        self, steps: list[Any], initial_text: str
    ) -> tuple[str, list[ToolCall], list[dict[str, Any]], list[str]]:
        text = initial_text
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
                        if val:
                            t_str += val
                elif isinstance(summary, str):
                    t_str = summary
                elif getattr(step, "text", None):
                    t_str = getattr(step, "text")

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
                if not text:
                    content = (
                        getattr(step, "content", None)
                        or (step.get("content") if isinstance(step, dict) else None)
                    )
                    if isinstance(content, list):
                        for c in content:
                            val = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                            if val:
                                text += val

        return text, tool_calls, thought_blocks, thought_parts

    def _parse_usage(self, usage_obj: Any) -> Any:
        if usage_obj is None:
            return None

        pt = getattr(usage_obj, "total_input_tokens", None)
        ct = getattr(usage_obj, "total_output_tokens", None)
        tt = getattr(usage_obj, "total_tokens", None)
        cached = getattr(usage_obj, "total_cached_tokens", 0) or 0
        reasoning = getattr(usage_obj, "total_thought_tokens", 0) or 0

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
