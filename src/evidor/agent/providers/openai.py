"""OpenAI adapter."""

import json
from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, ToolCall


def _safe_parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return {}
    try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict):
            return parsed
        return {"__raw_args__": parsed}
    except Exception as err:
        return {"__raw_args__": raw_args, "__decode_error__": str(err)}


class OpenAIProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    @property
    def model(self) -> str:
        return self._model

    def with_model(self, model: str) -> "OpenAIProvider":
        return OpenAIProvider(model=model, api_key=self._api_key)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("Install OpenAI support with: pip install 'evidor[openai]'") from error

        client = OpenAI(api_key=self._api_key)

        # Build tools if provided
        openai_tools: list[dict[str, Any]] | None = None
        if request.tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]

        # Use Responses API if available on client (e.g. experimental/mocked responses)
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            input_items: list[dict[str, Any]] = []
            for message in request.messages:
                if message.role == "tool":
                    input_items.append({
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    })
                elif message.tool_calls:
                    item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in message.tool_calls
                    ]
                    input_items.append(item)
                else:
                    input_items.append({"role": message.role, "content": message.content})

            kwargs: dict[str, Any] = {"model": self._model, "input": input_items}
            if openai_tools:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                    for t in request.tools
                ]
            response = client.responses.create(**kwargs)
            text = getattr(response, "output_text", None) or ""
            tool_calls: list[ToolCall] = []
            output_list = getattr(response, "output", None) or []
            for item in output_list:
                if getattr(item, "type", None) == "function_call":
                    args = _safe_parse_tool_arguments(item.arguments)
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", f"call_{len(tool_calls)+1}")
                    tool_calls.append(ToolCall(id=call_id, name=item.name, arguments=args))
            return GenerationResponse(text=text, model=self._model, tool_calls=tuple(tool_calls))

        # Standard Chat Completions API
        chat_messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                })
            elif message.tool_calls:
                msg_dict: dict[str, Any] = {"role": "assistant", "content": message.content or None}
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in message.tool_calls
                ]
                chat_messages.append(msg_dict)
            else:
                chat_messages.append({"role": message.role, "content": message.content})

        chat_kwargs: dict[str, Any] = {"model": self._model, "messages": chat_messages}
        if openai_tools:
            chat_kwargs["tools"] = openai_tools

        chat_response = client.chat.completions.create(**chat_kwargs)
        choice = chat_response.choices[0]
        choice_msg = choice.message
        text = choice_msg.content or ""
        tool_calls = []
        if getattr(choice_msg, "tool_calls", None):
            for tc in choice_msg.tool_calls:
                tc_args = _safe_parse_tool_arguments(tc.function.arguments)
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=tc_args))

        return GenerationResponse(text=text, model=self._model, tool_calls=tuple(tool_calls))
