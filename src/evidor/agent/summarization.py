"""Conversation summarization and summary-provider configuration."""

from collections.abc import Sequence

from ..models import GenerationRequest, Message
from ..providers.base import ModelProvider


SUMMARY_INSTRUCTION = (
    "Summarize the conversation below for a future assistant turn. Preserve user goals, "
    "decisions, constraints, factual details, and unresolved questions. Be concise."
)


class ConversationSummarizer:
    """Formats a transcript and asks a dedicated provider to summarize it."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def summarize(self, messages: Sequence[Message]) -> str:
        transcript_lines: list[str] = []
        for message in messages:
            if message.role == "tool":
                transcript_lines.append(f"TOOL ({message.name or 'unknown'}): {message.content}")
            elif message.tool_calls:
                call_names = ", ".join(call.name for call in message.tool_calls)
                prefix = f"ASSISTANT (called tools: {call_names}):"
                transcript_lines.append(f"{prefix} {message.content}" if message.content else prefix)
            else:
                transcript_lines.append(f"{message.role.upper()}: {message.content}")
        request = GenerationRequest(
            messages=(
                Message(role="system", content=SUMMARY_INSTRUCTION),
                Message(role="user", content="\n".join(transcript_lines)),
            )
        )
        return self.provider.generate(request).text


def resolve_summary_provider(
    provider: ModelProvider, summarization_model: str | ModelProvider | None
) -> ModelProvider:
    """Select a summary provider from a provider instance or model-name override."""
    if summarization_model is None:
        return provider
    if isinstance(summarization_model, ModelProvider):
        return summarization_model
    if isinstance(summarization_model, str):
        if not summarization_model.strip():
            raise ValueError("summarization_model must not be empty")
        with_model = getattr(provider, "with_model", None)
        if callable(with_model):
            return with_model(summarization_model)
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
