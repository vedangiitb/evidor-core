"""Google Gemini adapter."""

from evidor.models import GenerationRequest, GenerationResponse


class GeminiProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from google import genai
        except ImportError as error:
            raise ImportError("Install Gemini support with: pip install 'evidor[gemini]'") from error

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(model=self._model, contents=request.prompt)
        return GenerationResponse(text=response.text or "", model=self._model)
