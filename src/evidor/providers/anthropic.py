"""Anthropic adapter."""

from evidor.models import GenerationRequest, GenerationResponse


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from anthropic import Anthropic
        except ImportError as error:
            raise ImportError("Install Anthropic support with: pip install 'evidor[anthropic]'") from error

        client = Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": request.prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return GenerationResponse(text=text, model=self._model)
