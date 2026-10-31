"""Request payload builder and schema sanitizer for Google Gemini."""

from collections.abc import Sequence
from typing import Any

from evidor.models import GenerationRequest, Message
from ..base import extract_system_prompt, sort_tools

_SNAKE_TO_CAMEL: dict[str, str] = {
    "any_of": "anyOf",
    "max_length": "maxLength",
    "min_length": "minLength",
    "max_items": "maxItems",
    "min_items": "minItems",
    "max_properties": "maxProperties",
    "min_properties": "minProperties",
    "property_ordering": "propertyOrdering",
}

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
    "anyOf",
    "any_of",
    "maximum",
    "minimum",
    "maxLength",
    "max_length",
    "minLength",
    "min_length",
    "pattern",
    "maxItems",
    "max_items",
    "minItems",
    "min_items",
    "maxProperties",
    "max_properties",
    "minProperties",
    "min_properties",
    "example",
    "propertyOrdering",
    "property_ordering",
}


class GeminiPayloadBuilder:
    """Builds and formats API request payloads for Google Gemini Interactions API."""

    def __init__(
        self,
        model: str,
        cached_content: str | None = None,
        thinking: Any = None,
    ) -> None:
        self.model = model
        self.cached_content = cached_content
        self.thinking = thinking

    def _sanitize_schema(self, schema: Any) -> Any:
        """Sanitize a JSON schema dictionary for Google GenAI types.Schema."""
        if not isinstance(schema, dict):
            if isinstance(schema, list):
                return [self._sanitize_schema(item) for item in schema]
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
            if k in ("oneOf", "one_of") and "anyOf" not in cleaned:
                cleaned["anyOf"] = (
                    [self._sanitize_schema(it) for it in v]
                    if isinstance(v, list)
                    else self._sanitize_schema(v)
                )
                continue
            if k == "examples" and "example" not in schema and isinstance(v, list) and v:
                cleaned["example"] = v[0]
                continue

            target_k = _SNAKE_TO_CAMEL.get(k, k)

            if k == "properties" and isinstance(v, dict):
                cleaned["properties"] = {
                    prop_name: self._sanitize_schema(prop_s)
                    for prop_name, prop_s in v.items()
                }
            elif k == "items":
                if isinstance(v, list):
                    if len(v) == 1:
                        cleaned["items"] = self._sanitize_schema(v[0])
                    elif len(v) > 1:
                        cleaned["items"] = {"anyOf": [self._sanitize_schema(it) for it in v]}
                else:
                    cleaned["items"] = self._sanitize_schema(v)
            elif k in _ALLOWED_GEMINI_SCHEMA_KEYS:
                cleaned[target_k] = self._sanitize_schema(v)

        # Ensure ARRAY types have an items schema (required by Gemini API)
        if cleaned.get("type") in ("array", "ARRAY") and "items" not in cleaned:
            cleaned["items"] = {"type": "string"}

        # Omit empty required lists (Gemini schema validation rejects empty required arrays)
        if "required" in cleaned and not cleaned["required"]:
            del cleaned["required"]

        return cleaned

    def build(self, request: GenerationRequest, types: Any = None) -> dict[str, Any]:
        """Construct the full request dictionary for Gemini Interactions API."""
        system_prompt = extract_system_prompt(request)
        input_steps = self.format_input_steps(request.messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_steps,
        }

        if system_prompt:
            payload["system_instruction"] = system_prompt
        if request.previous_response_id:
            payload["previous_interaction_id"] = request.previous_response_id
        if request.store is not None:
            payload["store"] = request.store

        if request.tools:
            sorted_tools = sort_tools(request.tools)
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": self._sanitize_schema(t.parameters) if t.parameters else None,
                }
                for t in sorted_tools
            ]

        if self.thinking is not None:
            gen_config: dict[str, Any] = {}
            if not getattr(self.thinking, "enabled", True):
                gen_config["thinking_level"] = "minimal"
                gen_config["thinking_summaries"] = "none"
            else:
                thinking_level = getattr(self.thinking, "thinking_level", None)
                if thinking_level:
                    gen_config["thinking_level"] = thinking_level

                thinking_summaries = getattr(self.thinking, "thinking_summaries", None)
                if thinking_summaries:
                    gen_config["thinking_summaries"] = thinking_summaries

            if gen_config:
                payload["generation_config"] = gen_config

        return payload

    def format_input_steps(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Format domain messages into Gemini-compliant input steps."""
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
                thought_steps: list[dict[str, Any]] = []
                function_call_steps: list[dict[str, Any]] = []
                other_steps: list[Any] = []

                if message.thought_blocks:
                    for tb in message.thought_blocks:
                        if isinstance(tb, dict):
                            tb_type = tb.get("type")
                            sig = tb.get("thought_signature") or tb.get("thoughtSignature") or tb.get("signature")
                            if tb_type == "thought" or tb.get("thought"):
                                thought_steps.append({
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
                                function_call_steps.append(fc_step)
                            else:
                                other_steps.append(tb)
                        elif hasattr(tb, "type"):
                            other_steps.append(tb)

                if message.tool_calls:
                    existing_call_ids = {
                        step.get("id")
                        for step in function_call_steps
                        if isinstance(step, dict) and step.get("type") == "function_call"
                    }
                    for tc in message.tool_calls:
                        if tc.id not in existing_call_ids:
                            function_call_steps.append({
                                "type": "function_call",
                                "id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            })

                input_steps.extend(thought_steps)
                input_steps.extend(other_steps)
                if message.content:
                    input_steps.append({
                        "type": "model_output",
                        "content": [{"type": "text", "text": message.content}],
                    })
                input_steps.extend(function_call_steps)
            else:
                input_steps.append({
                    "type": "user_input",
                    "content": [{"type": "text", "text": message.content}],
                })
        return input_steps


