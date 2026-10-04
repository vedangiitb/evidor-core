"""Conversation message storage and recording helpers."""

from collections.abc import Sequence

from ...models import GenerationResponse, Message, ToolCall


class ConversationHistory:
    """Owns the mutable transcript for a single agent session."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, response: GenerationResponse) -> None:
        self.messages.append(Message(role="assistant", content=response.text, tool_calls=response.tool_calls))

    def add_tool_result(self, call: ToolCall, result: str) -> None:
        self.messages.append(Message(role="tool", content=result, tool_call_id=call.id, name=call.name))

    def replace(self, messages: Sequence[Message]) -> None:
        """Replace history in place so existing list references remain valid."""
        self.messages[:] = messages

    def clear(self) -> None:
        """Remove conversation turns while retaining the original system prompt."""
        self.messages[:] = [
            message for message in self.messages if message.role == "system" and not message.is_summary
        ]
