"""Provider-neutral data structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .tools import Tool


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A provider-neutral result returned by a web search provider."""

    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by a model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tool call id must not be empty")
        if not self.name:
            raise ValueError("tool call name must not be empty")


@dataclass(frozen=True, slots=True)
class Message:
    """A single text message or tool call/response in a conversation."""

    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    is_summary: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.tool_calls, (list, tuple)):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if (not self.content or not self.content.strip()) and not self.tool_calls and self.role != "tool":
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True, init=False)
class GenerationRequest:
    """The complete provider-neutral input for one model generation."""

    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]

    def __init__(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Message] | None = None,
        tools: Sequence[Tool] | None = None,
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
        object.__setattr__(self, "tools", tuple(tools or ()))

    @property
    def prompt(self) -> str:
        """Compatibility alias for the newest message content."""
        return self.messages[-1].content


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """The normalized output returned by a model provider."""

    text: str
    model: str
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.tool_calls, (list, tuple)):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
