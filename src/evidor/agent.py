import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from .context import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_MESSAGES, ConversationContext
from .models import GenerationRequest, GenerationResponse, Message, ToolCall
from .providers.base import ModelProvider
from .tools import Tool, tool
from .utils import format_tool_error, serialize_tool_result


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
        tool_timeout: float | None = None,
        system_prompt: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        summarization_model: str | ModelProvider | None = None,
    ) -> None:
        self._provider = provider
        self._summary_provider = self._resolve_summary_provider(provider, summarization_model)
        self._context = ConversationContext(context_window, max_messages)
        self._messages: list[Message] = []
        self._max_tool_iterations = max_tool_iterations
        self._tool_timeout = tool_timeout

        configured_tools: list[Tool] = []
        for t in (tools or ()):
            tool_obj = t if isinstance(t, Tool) else tool(t)
            if tool_timeout is not None and tool_obj.timeout is None:
                tool_obj = tool_obj.with_timeout(tool_timeout)
            configured_tools.append(tool_obj)
        self._tools: tuple[Tool, ...] = tuple(configured_tools)
        self._tool_map: dict[str, Tool] = {t.name: t for t in self._tools}

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
    def tool_timeout(self) -> float | None:
        """Default per-tool execution timeout in seconds, if configured."""
        return self._tool_timeout

    @property
    def summary_provider(self) -> ModelProvider:
        """The provider used for compacting and summarizing conversations."""
        return self._summary_provider

    def _resolve_tool(self, call: ToolCall) -> Tool | str:
        if "__decode_error__" in call.arguments:
            decode_err = call.arguments["__decode_error__"]
            raw_input = call.arguments.get("__raw_args__", "")
            return (
                f"Error: Malformed JSON arguments for tool '{call.name}': {decode_err}. "
                f"Received input: {raw_input}"
            )
        target_tool = self._tool_map.get(call.name)
        if target_tool is None:
            return f"Error: Tool '{call.name}' not found."
        return target_tool

    def _execute_tool_call(self, call: ToolCall) -> str:
        target = self._resolve_tool(call)
        if isinstance(target, str):
            return target
        try:
            return serialize_tool_result(target.execute(**call.arguments))
        except Exception as err:
            return format_tool_error(call.name, err)

    async def _execute_tool_call_async(self, call: ToolCall) -> str:
        target = self._resolve_tool(call)
        if isinstance(target, str):
            return target
        try:
            return serialize_tool_result(await target.execute_async(**call.arguments))
        except Exception as err:
            return format_tool_error(call.name, err)

    def _record_assistant(self, response: GenerationResponse) -> None:
        self._messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
            )
        )

    def _record_tool_result(self, call: ToolCall, result_str: str) -> None:
        self._messages.append(
            Message(
                role="tool",
                content=result_str,
                tool_call_id=call.id,
                name=call.name,
            )
        )

    def _execute_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        for call in calls:
            self._record_tool_result(call, self._execute_tool_call(call))

    async def _execute_tool_calls_async(self, calls: Sequence[ToolCall]) -> None:
        for call in calls:
            self._record_tool_result(call, await self._execute_tool_call_async(call))

    def _compact(self) -> None:
        self._messages = self._context.compact(self._messages, self._summarize)

    async def _compact_async(self) -> None:
        self._messages = await asyncio.to_thread(self._context.compact, self._messages, self._summarize)

    async def _generate_provider_response_async(self, request: GenerationRequest) -> GenerationResponse:
        return await asyncio.to_thread(self._provider.generate, request)

    def _finalize_loop_expiry(
        self,
        final_response: GenerationResponse,
        last_response: GenerationResponse,
    ) -> GenerationResponse:
        """Handle final synthesis and record the closing assistant message when max iterations is reached."""
        final_text = (final_response.text or "").strip()
        if not final_text:
            final_text = (
                (last_response.text or "").strip()
                or f"Reached maximum tool iterations ({self._max_tool_iterations}) without a final response."
            )

        final_msg = Message(role="assistant", content=final_text)
        self._messages.append(final_msg)
        return GenerationResponse(
            text=final_text,
            model=final_response.model or last_response.model,
        )

    def send(self, message: str) -> GenerationResponse:
        """Advance the conversation by one user turn and any resulting tool execution loops."""
        self._messages.append(Message(role="user", content=message))

        for _ in range(self._max_tool_iterations):
            self._compact()
            response = self._provider.generate(
                GenerationRequest(messages=self._messages, tools=self._tools)
            )
            self._record_assistant(response)
            if not response.tool_calls:
                return response
            self._execute_tool_calls(response.tool_calls)

        # If loop reached max iterations and the last turn was still requesting tool calls:
        self._compact()
        final_response = self._provider.generate(GenerationRequest(messages=self._messages, tools=()))
        return self._finalize_loop_expiry(final_response, response)

    async def send_async(self, message: str) -> GenerationResponse:
        """Advance the conversation asynchronously by one user turn and any resulting tool execution loops."""
        self._messages.append(Message(role="user", content=message))

        for _ in range(self._max_tool_iterations):
            await self._compact_async()
            response = await self._generate_provider_response_async(
                GenerationRequest(messages=self._messages, tools=self._tools)
            )
            self._record_assistant(response)
            if not response.tool_calls:
                return response
            await self._execute_tool_calls_async(response.tool_calls)

        # If loop reached max iterations and the last turn was still requesting tool calls:
        await self._compact_async()
        final_response = await self._generate_provider_response_async(
            GenerationRequest(messages=self._messages, tools=())
        )
        return self._finalize_loop_expiry(final_response, response)

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
