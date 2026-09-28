"""Stateful, provider-independent conversational agent."""

from collections.abc import Callable, Sequence
import json
from typing import Any

from .context import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_MESSAGES, ConversationContext
from .models import GenerationRequest, GenerationResponse, Message
from .providers.base import ModelProvider
from .tools import Tool, tool


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
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        max_tool_iterations: int = 10,
        system_prompt: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        summarization_model: str | ModelProvider | None = None,
    ) -> None:
        self._provider = provider
        self._summary_provider = self._resolve_summary_provider(provider, summarization_model)
        self._context = ConversationContext(context_window, max_messages)
        self._messages: list[Message] = []
        self._tools: tuple[Tool, ...] = tuple(t if isinstance(t, Tool) else tool(t) for t in (tools or ()))
        self._tool_map: dict[str, Tool] = {t.name: t for t in self._tools}
        self._max_tool_iterations = max_tool_iterations
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    @property
    def messages(self) -> tuple[Message, ...]:
        """The retained conversation, including any generated summary message."""
        return tuple(self._messages)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The tools configured on this agent."""
        return self._tools

    @property
    def summary_provider(self) -> ModelProvider:
        """The provider used for compacting and summarizing conversations."""
        return self._summary_provider

    def send(self, message: str) -> GenerationResponse:
        """Advance the conversation by one user turn and any resulting tool execution loops."""
        self._messages.append(Message(role="user", content=message))

        for _ in range(self._max_tool_iterations):
            self._messages = self._context.compact(self._messages, self._summarize)
            response = self._provider.generate(
                GenerationRequest(messages=self._messages, tools=self._tools)
            )
            self._messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:
                return response

            for call in response.tool_calls:
                target_tool = self._tool_map.get(call.name)
                if target_tool is None:
                    result_str = f"Error: Tool '{call.name}' not found."
                else:
                    try:
                        result = target_tool.execute(**call.arguments)
                        if isinstance(result, str):
                            result_str = result
                        elif isinstance(result, (dict, list, int, float, bool)):
                            result_str = json.dumps(result)
                        else:
                            result_str = str(result)
                    except Exception as err:
                        result_str = f"Error executing tool '{call.name}': {err}"

                self._messages.append(
                    Message(
                        role="tool",
                        content=result_str,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        return response

    def clear_history(self) -> None:
        """Clear the conversation while retaining the optional system prompt."""
        self._messages = [message for message in self._messages if message.role == "system" and not message.is_summary]

    def _summarize(self, messages: Sequence[Message]) -> str:
        transcript_lines: list[str] = []
        for message in messages:
            if message.role == "tool":
                transcript_lines.append(f"TOOL ({message.name or 'unknown'}): {message.content}")
            elif message.tool_calls:
                call_names = ", ".join(call.name for call in message.tool_calls)
                prefix = f"ASSISTANT (called tools: {call_names}):"
                if message.content:
                    transcript_lines.append(f"{prefix} {message.content}")
                else:
                    transcript_lines.append(prefix)
            else:
                transcript_lines.append(f"{message.role.upper()}: {message.content}")

        transcript = "\n".join(transcript_lines)
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
        if isinstance(summarization_model, str):  # type: ignore[unreachable]
            if not summarization_model.strip():
                raise ValueError("summarization_model must not be empty")
            with_model = getattr(provider, "with_model", None)
            if callable(with_model):
                return with_model(summarization_model)  # type: ignore[no-any-return]
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
