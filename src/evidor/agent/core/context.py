"""Stateful conversation and context tracking for agents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evidor.models import GenerationRequest, Message
from evidor.tools import Tool


class ConversationContext:
    """Manages conversation messages, system prompt, and provider context state.

    Adheres to the Single Responsibility Principle by encapsulating:
    - Message history storage and mutations
    - Server-side context chaining identifiers (e.g. previous_response_id, conversation_id)
    - Construction of GenerationRequest payloads for both stateful and stateless strategies
    """

    def __init__(
        self,
        strategy: str,
        system_prompt: str | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        store: bool | None = None,
        compact_threshold: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.system_prompt = system_prompt
        self.last_response_id = previous_response_id
        self.conversation_id = conversation_id
        self.store = store
        self.compact_threshold = compact_threshold

        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    @property
    def messages(self) -> tuple[Message, ...]:
        """An immutable view of recorded conversation messages."""
        return tuple(self._messages)

    def clear(self) -> None:
        """Clear conversation history and reset provider state identifiers."""
        self.last_response_id = None
        self._messages.clear()
        if self.system_prompt:
            self._messages.append(Message(role="system", content=self.system_prompt))

    def spawn_turn_context(self) -> ConversationContext:
        """Create a fresh isolated turn context with the same configuration."""
        return ConversationContext(
            strategy=self.strategy,
            system_prompt=self.system_prompt,
            previous_response_id=None,
            conversation_id=None,
            store=self.store,
            compact_threshold=self.compact_threshold,
        )

    def copy_from(self, other: ConversationContext) -> None:
        """Update messages and last_response_id from another turn context."""
        self._messages = list(other._messages)
        self.last_response_id = other.last_response_id

    def append_user_message(self, message: str) -> Message:
        """Record a new user message in the conversation history."""
        msg = Message(role="user", content=message)
        self._messages.append(msg)
        return msg

    def append_assistant_response(
        self,
        text: str | None,
        tool_calls: tuple[Any, ...] = (),
        thinking: str | None = None,
        thought_blocks: tuple[Any, ...] = (),
        compaction: Any = None,
    ) -> Message:
        """Record an assistant turn in the conversation history."""
        asst_content = text
        if (
            (not asst_content or not asst_content.strip())
            and not tool_calls
            and not thought_blocks
            and not compaction
        ):
            asst_content = "Done."

        msg = Message(
            role="assistant",
            content=asst_content,
            tool_calls=tool_calls,
            thinking=thinking,
            thought_blocks=thought_blocks,
            compaction=compaction,
        )
        self._messages.append(msg)
        return msg

    def append_tool_result(self, tool_call_id: str, name: str, content: str) -> Message:
        """Record a tool execution result in the conversation history."""
        msg = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
        self._messages.append(msg)
        return msg

    def append_tool_results(self, tool_messages: Sequence[Message]) -> None:
        """Record multiple tool execution results in the conversation history."""
        self._messages.extend(tool_messages)

    def build_generation_request(
        self,
        current_turn_messages: Sequence[Message],
        tools: Sequence[Tool],
        *,
        stream: bool = False,
    ) -> GenerationRequest:
        """Construct a GenerationRequest based on the active provider context strategy."""
        is_stateful_chain = self.strategy in ("response_chain", "interaction_chain")
        if is_stateful_chain:
            return GenerationRequest(
                messages=list(current_turn_messages),
                tools=tuple(tools),
                previous_response_id=self.last_response_id,
                conversation_id=self.conversation_id,
                store=self.store,
                system=self.system_prompt,
                stream=stream,
            )
        return GenerationRequest(
            messages=list(self._messages),
            tools=tuple(tools),
            system=self.system_prompt,
            stream=stream,
        )

    def build_final_request(
        self,
        current_turn_messages: Sequence[Message],
        *,
        stream: bool = False,
    ) -> GenerationRequest:
        """Construct the terminal GenerationRequest (without tools) when iterations exhaust."""
        is_stateful_chain = self.strategy in ("response_chain", "interaction_chain")
        if is_stateful_chain:
            return GenerationRequest(
                messages=list(current_turn_messages),
                tools=(),
                previous_response_id=self.last_response_id,
                conversation_id=self.conversation_id,
                store=self.store,
                system=self.system_prompt,
                stream=stream,
            )
        return GenerationRequest(
            messages=list(self._messages),
            tools=(),
            system=self.system_prompt,
            stream=stream,
        )

