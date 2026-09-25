"""OpenAI adapter."""

from evidor.models import GenerationRequest, GenerationResponse


class OpenAIProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("Install OpenAI support with: pip install 'evidor[openai]'") from error

        client = OpenAI(api_key=self._api_key)
        response = client.responses.create(model=self._model, input=request.prompt)
        return GenerationResponse(text=response.output_text, model=self._model)
