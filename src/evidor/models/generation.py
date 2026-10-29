"""Core messaging and generation data structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from evidor.tools import Tool

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
    thinking: str | None = None
    thought_blocks: tuple[Any, ...] = ()
    compaction: str | None = None
    is_error: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.tool_calls, (list, tuple)):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if isinstance(self.thought_blocks, (list, tuple)):
            object.__setattr__(self, "thought_blocks", tuple(self.thought_blocks))
        if (
            (not self.content or not self.content.strip())
            and not self.tool_calls
            and not self.thought_blocks
            and not self.compaction
            and self.role != "tool"
        ):
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True, init=False)
class GenerationRequest:
    """The complete provider-neutral input for one model generation."""

    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    previous_response_id: str | None
    conversation_id: str | None
    store: bool | None
    system: str | None
    stream: bool

    def __init__(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Message] | None = None,
        tools: Sequence[Tool] | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        store: bool | None = None,
        system: str | None = None,
        stream: bool = False,
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
        object.__setattr__(self, "previous_response_id", previous_response_id)
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "stream", bool(stream))

    @property
    def prompt(self) -> str:
        """Convenience property accessing the newest message content."""
        return self.messages[-1].content


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """The normalized output returned by a model provider."""

    text: str
    model: str
    id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Any = None
    thinking: str | None = None
    thought_blocks: tuple[Any, ...] = ()
    compaction: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.tool_calls, (list, tuple)):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if isinstance(self.thought_blocks, (list, tuple)):
            object.__setattr__(self, "thought_blocks", tuple(self.thought_blocks))

