"""Google Gemini adapter."""

from evidor.models import GenerationRequest, GenerationResponse


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
        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in request.messages
            if message.role != "system"
        ]
        client = genai.Client(api_key=self._api_key)
        config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
        response = client.models.generate_content(model=self._model, contents=contents, config=config)
        return GenerationResponse(text=response.text or "", model=self._model)
