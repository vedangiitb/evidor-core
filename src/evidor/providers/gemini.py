"""Google Gemini adapter."""

from typing import Any

from evidor.models import GenerationRequest, GenerationResponse, ToolCall


class GeminiProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    @property
    def model(self) -> str:
        return self._model

    def with_model(self, model: str) -> "GeminiProvider":
        return GeminiProvider(model=model, api_key=self._api_key)

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
            declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
                for t in request.tools
            ]
            gemini_tools = [types.Tool(function_declarations=declarations)]

        config = None
        if system_prompt or gemini_tools:
            config_kwargs: dict[str, Any] = {}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools
            config = types.GenerateContentConfig(**config_kwargs)

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(model=self._model, contents=contents, config=config)

        # Parse text and tool calls
        text = getattr(response, "text", None) or ""
        tool_calls: list[ToolCall] = []

        # Check function_calls on response object
        if getattr(response, "function_calls", None):
            for i, fc in enumerate(response.function_calls):
                call_id = getattr(fc, "id", None) or f"call_{i+1}"
                args = dict(fc.args) if getattr(fc, "args", None) else {}
                tool_calls.append(ToolCall(id=call_id, name=fc.name, arguments=args))
        elif getattr(response, "candidates", None) and response.candidates:
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    call_id = getattr(fc, "id", None) or f"call_{len(tool_calls)+1}"
                    args = dict(fc.args) if getattr(fc, "args", None) else {}
                    tool_calls.append(ToolCall(id=call_id, name=fc.name, arguments=args))

        return GenerationResponse(text=text, model=self._model, tool_calls=tuple(tool_calls))
