import asyncio
from collections.abc import Awaitable, Callable, Sequence

from ...models import GenerationRequest, GenerationResponse, Message
from ..providers.base import ModelProvider


SUMMARY_INSTRUCTION = (
    "Summarize the conversation below for a future assistant turn. Preserve user goals, "
    "decisions, constraints, factual details, and unresolved questions. Be concise."
)


class ConversationSummarizer:
    """Formats a transcript and asks a dedicated provider to summarize it."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        generate_fn: Callable[[GenerationRequest], GenerationResponse] | None = None,
        generate_async_fn: Callable[[GenerationRequest], Awaitable[GenerationResponse]] | None = None,
    ) -> None:
        self.provider = provider
        self._generate_fn = generate_fn
        self._generate_async_fn = generate_async_fn

    def _build_request(self, messages: Sequence[Message]) -> GenerationRequest:
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
        return GenerationRequest(
            messages=(
                Message(role="system", content=SUMMARY_INSTRUCTION),
                Message(role="user", content="\n".join(transcript_lines)),
            )
        )

    def summarize(self, messages: Sequence[Message]) -> str:
        request = self._build_request(messages)
        if self._generate_fn is not None:
            return self._generate_fn(request).text
        return self.provider.generate(request).text

    async def summarize_async(self, messages: Sequence[Message]) -> str:
        request = self._build_request(messages)
        if self._generate_async_fn is not None:
            response = await self._generate_async_fn(request)
            return response.text
        if self._generate_fn is not None:
            response = await asyncio.to_thread(self._generate_fn, request)
            return response.text
        response = await asyncio.to_thread(self.provider.generate, request)
        return response.text


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
