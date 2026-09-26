"""Stateful, provider-independent conversational agent."""

from collections.abc import Sequence

from .context import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_MESSAGES, ConversationContext
from .models import GenerationRequest, GenerationResponse, Message
from .providers.base import ModelProvider


_SUMMARY_INSTRUCTION = (
    "Summarize the conversation below for a future assistant turn. Preserve user goals, "
    "decisions, constraints, factual details, and unresolved questions. Be concise."
)


class Agent:
    """A single conversational session backed by one model provider."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        system_prompt: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        summarization_model: str | ModelProvider | None = None,
    ) -> None:
        self._provider = provider
        self._summary_provider = self._resolve_summary_provider(provider, summarization_model)
        self._context = ConversationContext(context_window, max_messages)
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    @property
    def messages(self) -> tuple[Message, ...]:
        """The retained conversation, including any generated summary message."""
        return tuple(self._messages)

    @property
    def summary_provider(self) -> ModelProvider:
        """The provider used for compacting and summarizing conversations."""
        return self._summary_provider

    def send(self, message: str) -> GenerationResponse:
        """Advance the conversation by one user/assistant turn."""
        self._messages.append(Message(role="user", content=message))
        self._messages = self._context.compact(self._messages, self._summarize)
        response = self._provider.generate(GenerationRequest(messages=self._messages))
        self._messages.append(Message(role="assistant", content=response.text))
        return response

    def clear_history(self) -> None:
        """Clear the conversation while retaining the optional system prompt."""
        self._messages = [message for message in self._messages if message.role == "system" and not message.is_summary]

    def _summarize(self, messages: Sequence[Message]) -> str:
        transcript = "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)
        request = GenerationRequest(
            messages=(
                Message(role="system", content=_SUMMARY_INSTRUCTION),
                Message(role="user", content=transcript),
            )
        )
        return self._summary_provider.generate(request).text

    @staticmethod
    def _resolve_summary_provider(
        provider: ModelProvider,
        summarization_model: str | ModelProvider | None,
    ) -> ModelProvider:
        if summarization_model is None:
            return provider
        if isinstance(summarization_model, ModelProvider):
            return summarization_model
        if isinstance(summarization_model, str): # type: ignore
            if not summarization_model.strip():
                raise ValueError("summarization_model must not be empty")
            with_model = getattr(provider, "with_model", None)
            if callable(with_model):
                return with_model(summarization_model) # type: ignore
            provider_type = type(provider)
            api_key = getattr(provider, "_api_key", None)
            try:
                return provider_type(model=summarization_model, api_key=api_key)  # type: ignore[call-arg]
            except TypeError:
                try:
                    return provider_type(model=summarization_model)  # type: ignore[call-arg]
                except TypeError as err:
                    raise ValueError(
                        f"Cannot configure summarization_model string '{summarization_model}' for provider {provider_type.__name__}. "
                        "Pass a ModelProvider instance or implement with_model(model)."
                    ) from err
        raise TypeError(
            f"summarization_model must be a model name string or ModelProvider, got {type(summarization_model).__name__}"
        )
