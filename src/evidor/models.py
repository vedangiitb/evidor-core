"""Provider-neutral data structures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """The complete input for a single model generation."""

    prompt: str

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """The normalized output returned by a model provider."""

    text: str
    model: str
