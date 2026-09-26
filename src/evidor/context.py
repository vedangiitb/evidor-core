"""Conversation-window management and summarization."""

from collections.abc import Callable, Sequence

from .models import Message

DEFAULT_CONTEXT_WINDOW = 16_000
DEFAULT_MAX_MESSAGES = 50
_CHARS_PER_TOKEN = 4


class ConversationContext:
    """Keeps recent messages and replaces older messages with a bounded summary."""

    def __init__(self, context_window: int, max_messages: int) -> None:
        if context_window < 128:
            raise ValueError("context_window must be at least 128 tokens")
        if max_messages < 3:
            raise ValueError("max_messages must be at least 3")
        self._context_window = context_window
        self._max_messages = max_messages

    def compact(
        self,
        messages: Sequence[Message],
        summarize: Callable[[Sequence[Message]], str],
    ) -> list[Message]:
        """Return messages that fit the configured input budget."""
        if not self._requires_compaction(messages):
            return list(messages)

        permanent = [message for message in messages if message.role == "system" and not message.is_summary]
        existing_summaries = [message for message in messages if message.is_summary]
        conversation = [message for message in messages if message.role != "system"]
        recent: list[Message] = list(conversation)
        summarized: list[Message] = list(existing_summaries)

        recent_limit = max(1, self._max_messages - len(permanent) - 1)
        while len(recent) > recent_limit or self._estimated_tokens(permanent + recent) > self._input_budget:
            if len(recent) == 1:
                break
            summarized.append(recent.pop(0))

        if not summarized:
            return permanent + recent

        summary = summarize(summarized).strip()
        if not summary:
            summary = "Earlier conversation was omitted because no summary was returned."
        summary = summary[: self._summary_token_limit * _CHARS_PER_TOKEN]
        return permanent + [Message(role="system", content=f"Conversation summary:\n{summary}", is_summary=True)] + recent

    @property
    def _input_budget(self) -> int:
        return self._context_window - min(1_024, self._context_window // 4)

    @property
    def _summary_token_limit(self) -> int:
        return min(1_024, max(32, self._context_window // 4))

    def _requires_compaction(self, messages: Sequence[Message]) -> bool:
        return len(messages) > self._max_messages or self._estimated_tokens(messages) > self._input_budget

    @staticmethod
    def _estimated_tokens(messages: Sequence[Message]) -> int:
        return sum((len(message.content) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN + 4 for message in messages)
