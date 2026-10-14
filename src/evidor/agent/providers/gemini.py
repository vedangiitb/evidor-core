"""Google Gemini adapter."""

from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, ToolCall


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
    """Sanitize a JSON schema dictionary for Google GenAI types.Schema.

    google-genai and Gemini API protobuf support only a strict subset of JSON Schema:
    - Vendor extensions (x-*) and meta-schema keywords ($schema, $id, $comment) are stripped.
    - Unsupported fields like additionalProperties and defs are stripped (unrecognized by Gemini API proto).
    - Multi-type lists (e.g. type: ['string', 'number', 'boolean']) are converted to anyOf.
    - const is converted to enum: [const].
    - oneOf is converted to anyOf.
    - items as a list is unwrapped or converted to anyOf.
    - Property definitions in 'properties' are sanitized recursively.
    """
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


class GeminiProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cached_content: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._cached_content = cached_content

    @property
    def model(self) -> str:
        return self._model

    @property
    def cached_content(self) -> str | None:
        return self._cached_content

    def with_model(self, model: str) -> "GeminiProvider":
        return GeminiProvider(
            model=model,
            api_key=self._api_key,
            cached_content=self._cached_content,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise ImportError("Install Gemini support with: pip install 'evidor[gemini]'") from error

        system_prompt = "\n\n".join(message.content for message in request.messages if message.role == "system")
        contents: list[Any] = []

        for message in request.messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                part = types.Part.from_function_response(
                    name=message.name or "tool",
                    response={"result": message.content},
                )
                contents.append(types.Content(role="user", parts=[part]))
            elif message.role == "assistant":
                parts: list[Any] = []
                if message.content:
                    parts.append(types.Part.from_text(text=message.content))
                if message.tool_calls:
                    for tc in message.tool_calls:
                        parts.append(types.Part.from_function_call(name=tc.name, args=tc.arguments))
                contents.append(types.Content(role="model", parts=parts))
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message.content)],
                    )
                )

        gemini_tools: list[Any] | None = None
        if request.tools:
            sorted_tools = sorted(request.tools, key=lambda t: t.name)
            declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=_sanitize_gemini_schema(t.parameters, types) if t.parameters else None,
                )
                for t in sorted_tools
            ]
            gemini_tools = [types.Tool(function_declarations=declarations)]

        config = None
        if system_prompt or gemini_tools or self._cached_content:
            config_kwargs: dict[str, Any] = {}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools
            if self._cached_content:
                config_kwargs["cached_content"] = self._cached_content
            config = types.GenerateContentConfig(**config_kwargs)

        http_options = None
        http_options_cls = getattr(types, "HttpOptions", None)
        retry_options_cls = getattr(types, "HttpRetryOptions", None)
        if http_options_cls is not None and retry_options_cls is not None:
            http_options = http_options_cls(retry_options=retry_options_cls(attempts=1))
        elif http_options_cls is not None:
            http_options = http_options_cls(retry_options={"attempts": 1})
        else:
            http_options = {"retry_options": {"attempts": 1}}

        client = genai.Client(api_key=self._api_key, http_options=http_options)
        response = client.models.generate_content(model=self._model, contents=contents, config=config)

        # Parse text and tool calls.
        # Extract directly from candidate parts to avoid triggering google-genai's
        # warning when non-text parts (e.g. function calls) are present.
        text = ""
        tool_calls: list[ToolCall] = []

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, (list, tuple)) and candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            text_chunks: list[str] = []
            thought_chunks: list[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                is_thought = getattr(part, "thought", False)
                if isinstance(part_text, str):
                    if not is_thought:
                        text_chunks.append(part_text)
                    else:
                        thought_chunks.append(part_text)

                fc = getattr(part, "function_call", None)
                if fc:
                    call_id = getattr(fc, "id", None) or f"call_{len(tool_calls)+1}"
                    args = dict(fc.args) if getattr(fc, "args", None) else {}
                    tool_calls.append(ToolCall(id=call_id, name=fc.name or "", arguments=args))

            if text_chunks:
                text = "".join(text_chunks)
            elif not tool_calls and thought_chunks:
                text = "".join(thought_chunks)

        # Fallback for tool_calls if not extracted from candidates' parts
        if not tool_calls and getattr(response, "function_calls", None):
            raw_fc = response.function_calls
            if isinstance(raw_fc, (list, tuple)):
                for i, fc in enumerate(raw_fc):
                    call_id = getattr(fc, "id", None) or f"call_{i+1}"
                    args = dict(fc.args) if getattr(fc, "args", None) else {}
                    tool_calls.append(ToolCall(id=call_id, name=fc.name or "", arguments=args))

        # Fallback for text if response lacks candidate parts (e.g. mock objects)
        if not text and not (isinstance(candidates, (list, tuple)) and candidates):
            raw_text = getattr(response, "text", None)
            if isinstance(raw_text, str):
                text = raw_text

        # Extract token usage metadata if available
        usage = None
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            try:
                from evidor.telemetry.events import TokenUsage

                pt = getattr(usage_metadata, "prompt_token_count", None)
                ct = getattr(usage_metadata, "candidates_token_count", None)
                tt = getattr(usage_metadata, "total_token_count", None)
                cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0) or 0
                if isinstance(pt, int) or isinstance(ct, int) or isinstance(tt, int):
                    prompt_tokens = int(pt) if isinstance(pt, int) else 0
                    completion_tokens = int(ct) if isinstance(ct, int) else 0
                    total_tokens = int(tt) if isinstance(tt, int) else (prompt_tokens + completion_tokens)
                    usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=int(cached_tokens) if isinstance(cached_tokens, int) else 0,
                    )
            except Exception:
                pass

        return GenerationResponse(text=text, model=self._model, tool_calls=tuple(tool_calls), usage=usage)

