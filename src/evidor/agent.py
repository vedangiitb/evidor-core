"""Application service for one input-to-output interaction."""

from .models import GenerationRequest, GenerationResponse
from .providers.base import ModelProvider


class Agent:
    """A provider-independent entry point for generating a response."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    def run(self, prompt: str) -> GenerationResponse:
        """Send a prompt to the configured provider and return its response."""
        return self._provider.generate(GenerationRequest(prompt=prompt))
