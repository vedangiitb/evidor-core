"""Provider-neutral data structures."""

from dataclasses import dataclass
from typing import Literal, Sequence


MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """A single text message in a conversation."""

    role: MessageRole
    content: str
    is_summary: bool = False

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True, init=False)
class GenerationRequest:
    """The complete provider-neutral input for one model generation."""

    messages: tuple[Message, ...]

    def __init__(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Message] | None = None,
    ) -> None:
        if prompt is not None and messages is not None:
            raise ValueError("provide either prompt or messages, not both")
        if messages is None:
            if prompt is None or not prompt.strip():
                raise ValueError("prompt must not be empty")
            messages = (Message(role="user", content=prompt),)
        if not messages:
            raise ValueError("messages must not be empty")
        object.__setattr__(self, "messages", tuple(messages))

    @property
    def prompt(self) -> str:
        """Compatibility alias for the newest message content."""
        return self.messages[-1].content


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """The normalized output returned by a model provider."""

    text: str
    model: str
