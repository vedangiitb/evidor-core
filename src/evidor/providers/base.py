"""The port implemented by every model-provider adapter."""

from typing import Protocol, runtime_checkable

from evidor.models import GenerationRequest, GenerationResponse


@runtime_checkable
class ModelProvider(Protocol):
    """Produces one normalized response from one normalized request."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the request."""
        ...
