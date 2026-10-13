"""Conversation-window management and summarization."""

from collections.abc import Callable, Sequence
from typing import Any

from ...models import Message

DEFAULT_CONTEXT_WINDOW = 32_000
DEFAULT_MAX_MESSAGES = 50
DEFAULT_HIGH_WATER_RATIO = 0.85
DEFAULT_LOW_WATER_RATIO = 0.45
DEFAULT_CHAIN_MILESTONES = True
_CHARS_PER_TOKEN = 4


class ConversationContext:
    """Keeps recent messages and manages bounded summaries with generational hysteresis."""

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        *,
        high_water_ratio: float = DEFAULT_HIGH_WATER_RATIO,
        low_water_ratio: float = DEFAULT_LOW_WATER_RATIO,
        chain_milestones: bool = DEFAULT_CHAIN_MILESTONES,
    ) -> None:
        if context_window < 128:
            raise ValueError("context_window must be at least 128 tokens")
        if max_messages < 3:
            raise ValueError("max_messages must be at least 3")
        if not (0.0 < low_water_ratio < high_water_ratio <= 1.0):
            raise ValueError("low_water_ratio must be less than high_water_ratio and both between 0 and 1")
        self._context_window = context_window
        self._max_messages = max_messages
        self._high_water_ratio = high_water_ratio
        self._low_water_ratio = low_water_ratio
        self._chain_milestones = chain_milestones

    @property
    def chain_milestones(self) -> bool:
        return self._chain_milestones

    @property
    def high_water_ratio(self) -> float:
        return self._high_water_ratio

    @property
    def low_water_ratio(self) -> float:
        return self._low_water_ratio

    def compact(
        self,
        messages: Sequence[Message],
        summarize: Callable[[Sequence[Message]], str],
    ) -> list[Message]:
        """Return messages that fit the configured input budget using hysteresis batch pruning."""
        if not self._requires_compaction(messages):
            return list(messages)

        permanent = [message for message in messages if message.role == "system" and not message.is_summary]
        existing_summaries = [message for message in messages if message.is_summary]
        conversation = [message for message in messages if message.role != "system"]
        recent: list[Message] = list(conversation)

        if self._chain_milestones:
            # Preserve existing milestone blocks as immutable prefix
            summarized: list[Message] = []
            retained_summaries = existing_summaries
        else:
            # Re-synthesize existing summaries into a unified summary block
            summarized = list(existing_summaries)
            retained_summaries = []

        # Prune batch down to the low-water mark to create an append-only stability window
        target_recent_messages = max(
            1,
            self._low_water_messages - len(permanent) - len(retained_summaries),
        )
        target_token_budget = self._low_water_token_budget

        while (
            len(recent) > target_recent_messages
            or self._estimated_tokens(permanent + retained_summaries + recent) > target_token_budget
        ):
            if len(recent) == 1:
                break
            summarized.append(recent.pop(0))

        if not summarized:
            return permanent + retained_summaries + recent

        summary = summarize(summarized).strip()
        if not summary:
            summary = "Earlier conversation was omitted because no summary was returned."
        summary = summary[: self._summary_token_limit * _CHARS_PER_TOKEN]

        new_summary = Message(
            role="system",
            content=f"Conversation summary:\n{summary}",
            is_summary=True,
        )
        return permanent + retained_summaries + [new_summary] + recent

    async def compact_async(
        self,
        messages: Sequence[Message],
        summarize_async: Callable[[Sequence[Message]], Any],
    ) -> list[Message]:
        """Asynchronously return messages that fit the configured input budget using hysteresis batch pruning."""
        if not self._requires_compaction(messages):
            return list(messages)

        permanent = [message for message in messages if message.role == "system" and not message.is_summary]
        existing_summaries = [message for message in messages if message.is_summary]
        conversation = [message for message in messages if message.role != "system"]
        recent: list[Message] = list(conversation)

        if self._chain_milestones:
            summarized: list[Message] = []
            retained_summaries = existing_summaries
        else:
            summarized = list(existing_summaries)
            retained_summaries = []

        target_recent_messages = max(
            1,
            self._low_water_messages - len(permanent) - len(retained_summaries),
        )
        target_token_budget = self._low_water_token_budget

        while (
            len(recent) > target_recent_messages
            or self._estimated_tokens(permanent + retained_summaries + recent) > target_token_budget
        ):
            if len(recent) == 1:
                break
            summarized.append(recent.pop(0))

        if not summarized:
            return permanent + retained_summaries + recent

        raw_summary = await summarize_async(summarized)
        summary = raw_summary.strip() if isinstance(raw_summary, str) else str(raw_summary).strip()
        if not summary:
            summary = "Earlier conversation was omitted because no summary was returned."
        summary = summary[: self._summary_token_limit * _CHARS_PER_TOKEN]

        new_summary = Message(
            role="system",
            content=f"Conversation summary:\n{summary}",
            is_summary=True,
        )
        return permanent + retained_summaries + [new_summary] + recent

    @property
    def _high_water_token_budget(self) -> int:
        return max(32, int(self._context_window * self._high_water_ratio))

    @property
    def _low_water_token_budget(self) -> int:
        return max(16, int(self._context_window * self._low_water_ratio))

    @property
    def _input_budget(self) -> int:
        return self._high_water_token_budget

    @property
    def _high_water_messages(self) -> int:
        return self._max_messages

    @property
    def _low_water_messages(self) -> int:
        return max(1, int(self._max_messages * self._low_water_ratio))

    @property
    def _summary_token_limit(self) -> int:
        return min(1_024, max(32, self._context_window // 4))

    def _requires_compaction(self, messages: Sequence[Message]) -> bool:
        return len(messages) > self._high_water_messages or self._estimated_tokens(messages) > self._high_water_token_budget

    @staticmethod
    def _estimated_tokens(messages: Sequence[Message]) -> int:
        return sum((len(message.content) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN + 4 for message in messages)
