"""The public conversational agent facade."""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from ..context import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_MESSAGES, ConversationContext
from ..models import GenerationRequest, GenerationResponse, Message, ToolCall
from ..providers.base import ModelProvider
from ..tools import Tool
from .history import ConversationHistory
from .summarization import ConversationSummarizer, resolve_summary_provider
from .tool_executor import ToolExecutor


class Agent:
    """A conversational session coordinating a provider, history, context, and tools."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        max_tool_iterations: int = 10,
        tool_timeout: float | None = None,
        system_prompt: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        summarization_model: str | ModelProvider | None = None,
    ) -> None:
        self._provider = provider
        self._summary_provider = resolve_summary_provider(provider, summarization_model)
        self._summarizer = ConversationSummarizer(self._summary_provider)
        self._context = ConversationContext(context_window, max_messages)
        self._history = ConversationHistory(system_prompt)
        # Retained for compatibility with callers that inspect or seed test history.
        self._messages = self._history.messages
        self._max_tool_iterations = max_tool_iterations
        self._tool_timeout = tool_timeout
        self._tool_executor = ToolExecutor(tools, tool_timeout)
        self._tools = self._tool_executor.tools

    @property
    def messages(self) -> tuple[Message, ...]:
        """The retained conversation, including any generated summary message."""
        return tuple(self._messages)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The tools configured on this agent."""
        return self._tools

    @property
    def tool_timeout(self) -> float | None:
        """Default per-tool execution timeout in seconds, if configured."""
        return self._tool_timeout

    @property
    def summary_provider(self) -> ModelProvider:
        """The provider used for compacting and summarizing conversations."""
        return self._summary_provider

    def send(self, message: str) -> GenerationResponse:
        """Advance one user turn and execute any requested tool calls."""
        self._history.add_user(message)
        for _ in range(self._max_tool_iterations):
            self._compact()
            response = self._provider.generate(GenerationRequest(messages=self._messages, tools=self._tools))
            self._history.add_assistant(response)
            if not response.tool_calls:
                return response
            self._execute_tool_calls(response.tool_calls)
        self._compact()
        final_response = self._provider.generate(GenerationRequest(messages=self._messages, tools=()))
        return self._finalize_loop_expiry(final_response, response)

    async def send_async(self, message: str) -> GenerationResponse:
        """Asynchronously advance one user turn and execute requested tool calls."""
        self._history.add_user(message)
        for _ in range(self._max_tool_iterations):
            await self._compact_async()
            response = await asyncio.to_thread(
                self._provider.generate, GenerationRequest(messages=self._messages, tools=self._tools)
            )
            self._history.add_assistant(response)
            if not response.tool_calls:
                return response
            await self._execute_tool_calls_async(response.tool_calls)
        await self._compact_async()
        final_response = await asyncio.to_thread(
            self._provider.generate, GenerationRequest(messages=self._messages, tools=())
        )
        return self._finalize_loop_expiry(final_response, response)

    def clear_history(self) -> None:
        """Clear the conversation while retaining the optional system prompt."""
        self._history.clear()

    def _compact(self) -> None:
        self._history.replace(self._context.compact(self._messages, self._summarizer.summarize))

    async def _compact_async(self) -> None:
        compacted = await asyncio.to_thread(self._context.compact, self._messages, self._summarizer.summarize)
        self._history.replace(compacted)

    def _execute_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        for call in calls:
            self._history.add_tool_result(call, self._tool_executor.execute(call))

    async def _execute_tool_calls_async(self, calls: Sequence[ToolCall]) -> None:
        for call in calls:
            result = await self._tool_executor.execute_async(call)
            self._history.add_tool_result(call, result)

    def _finalize_loop_expiry(
        self, final_response: GenerationResponse, last_response: GenerationResponse
    ) -> GenerationResponse:
        final_text = (final_response.text or "").strip() or (last_response.text or "").strip()
        if not final_text:
            final_text = f"Reached maximum tool iterations ({self._max_tool_iterations}) without a final response."
        self._messages.append(Message(role="assistant", content=final_text))
        return GenerationResponse(text=final_text, model=final_response.model or last_response.model)

