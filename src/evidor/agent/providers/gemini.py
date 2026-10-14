"""Google Gemini adapter."""

from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, ToolCall


def _sanitize_gemini_schema(schema: Any, types_mod: Any) -> Any:
    """Sanitize a JSON schema dictionary for Google GenAI types.Schema.

    google-genai Schema models use extra='forbid'. Remove vendor extensions (e.g. x-*)
    and unsupported meta-schema keys ($schema, $id, $comment) to avoid ValidationError.
    """
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_sanitize_gemini_schema(item, types_mod) for item in schema]
        return schema

    allowed_keys = getattr(types_mod, "_ALLOWED_SCHEMA_KEYS", None)
    if allowed_keys is None:
        schema_model = getattr(types_mod, "Schema", None)
        if schema_model and hasattr(schema_model, "model_fields"):
            allowed_keys = set(schema_model.model_fields.keys())
            allowed_keys.update(f.alias for f in schema_model.model_fields.values() if f.alias)
        else:
            allowed_keys = {
                "type", "format", "title", "description", "default", "nullable",
                "enum", "properties", "required", "items", "all_of", "allOf",
                "any_of", "anyOf", "one_of", "oneOf", "not",
                "maximum", "minimum", "exclusive_maximum", "exclusiveMaximum",
                "exclusive_minimum", "exclusiveMinimum",
                "max_length", "maxLength", "min_length", "minLength",
                "pattern", "max_items", "maxItems", "min_items", "minItems",
                "unique_items", "uniqueItems", "max_properties", "maxProperties",
                "min_properties", "minProperties", "additional_properties", "additionalProperties",
                "defs", "$defs", "ref", "$ref", "example", "property_ordering", "propertyOrdering"
            }
        try:
            types_mod._ALLOWED_SCHEMA_KEYS = allowed_keys
        except Exception:
            pass

    cleaned: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "$ref":
            k = "ref"
        elif k in ("$defs", "definitions"):
            k = "defs"
        elif k == "examples" and "example" not in schema and isinstance(v, list) and v:
            cleaned["example"] = v[0]
            continue

        if k == "properties" and isinstance(v, dict):
            cleaned["properties"] = {
                prop_name: _sanitize_gemini_schema(prop_s, types_mod)
                for prop_name, prop_s in v.items()
            }
        elif k == "defs" and isinstance(v, dict):
            cleaned["defs"] = {
                d_name: _sanitize_gemini_schema(d_s, types_mod)
                for d_name, d_s in v.items()
            }
        elif k in allowed_keys:
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

