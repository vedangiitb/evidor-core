"""Request payload builder and schema sanitizer for Google Gemini."""

from collections.abc import Sequence
from typing import Any

from evidor.models import GenerationRequest, Message
from ..base import extract_system_prompt, sort_tools

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


def sanitize_gemini_schema(schema: Any, types_mod: Any) -> Any:
    """Sanitize a JSON schema dictionary for Google GenAI types.Schema."""
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [sanitize_gemini_schema(item, types_mod) for item in schema]
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
                [sanitize_gemini_schema(it, types_mod) for it in v]
                if isinstance(v, list)
                else sanitize_gemini_schema(v, types_mod)
            )
            continue
        if k == "examples" and "example" not in schema and isinstance(v, list) and v:
            cleaned["example"] = v[0]
            continue

        if k == "properties" and isinstance(v, dict):
            cleaned["properties"] = {
                prop_name: sanitize_gemini_schema(prop_s, types_mod)
                for prop_name, prop_s in v.items()
            }
        elif k == "items":
            if isinstance(v, list):
                if len(v) == 1:
                    cleaned["items"] = sanitize_gemini_schema(v[0], types_mod)
                elif len(v) > 1:
                    cleaned["items"] = {"anyOf": [sanitize_gemini_schema(it, types_mod) for it in v]}
            else:
                cleaned["items"] = sanitize_gemini_schema(v, types_mod)
        elif k in _ALLOWED_GEMINI_SCHEMA_KEYS:
            cleaned[k] = sanitize_gemini_schema(v, types_mod)

    return cleaned


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

    def build(self, request: GenerationRequest, types: Any) -> dict[str, Any]:
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
        if self.cached_content:
            payload["cached_content"] = self.cached_content

        if request.tools:
            sorted_tools = sort_tools(request.tools)
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": sanitize_gemini_schema(t.parameters, types) if t.parameters else None,
                }
                for t in sorted_tools
            ]

        if self.thinking is not None:
            gen_config: dict[str, Any] = {}
            if self.thinking.enabled:
                thinking_level = getattr(self.thinking, "thinking_level", None)
                if thinking_level:
                    gen_config["thinking_level"] = thinking_level
                elif self.thinking.thinking_budget is not None:
                    gen_config["thinking_level"] = "medium"
                if getattr(self.thinking, "include_thoughts", True):
                    gen_config["thinking_summaries"] = "auto"
            else:
                gen_config["thinking_level"] = "minimal"
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
                        step.get("id")
                        for step in input_steps
                        if isinstance(step, dict) and step.get("type") == "function_call"
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

