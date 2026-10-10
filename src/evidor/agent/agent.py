"""The public conversational agent facade."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
import secrets
import time
from typing import Any

from ..models import GenerationRequest, GenerationResponse, Message, ToolCall
from ..retry import DEFAULT_RETRY_CONFIG, RetryConfig, calculate_backoff_delay
from ..telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    TelemetryRuntime,
    TelemetrySink,
    TraceContext,
    create_child_trace_context,
    emit_event,
    get_current_trace_context,
    telemetry_scope,
    trace_scope,
)
from ..tools import Tool
from ..tools.core.tool_executor import ToolExecutor
from .context.context import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_MESSAGES, ConversationContext
from .context.history import ConversationHistory
from .context.summarization import ConversationSummarizer, resolve_summary_provider
from .providers.base import ModelProvider


class Agent:
    """A conversational session coordinating a provider, history, context, tools, and telemetry."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        mcp_servers: Any = None,
        mcp_client: Any = None,
        max_tool_iterations: int = 10,
        tool_timeout: float | None = None,
        system_prompt: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        summarization_model: str | ModelProvider | None = None,
        telemetry: TelemetryRuntime | TelemetrySink | Sequence[TelemetrySink] | None = None,
        retry_config: RetryConfig | bool | None = DEFAULT_RETRY_CONFIG,
        max_retries: int | None = None,
    ) -> None:

        self._provider = provider
        self._summary_provider = resolve_summary_provider(provider, summarization_model)
        self._summarizer = ConversationSummarizer(self._summary_provider)
        self._context = ConversationContext(context_window, max_messages)
        self._system_prompt = system_prompt
        self._history = ConversationHistory(system_prompt)
        # Retained for compatibility with callers that inspect or seed test history.
        self._messages = self._history.messages
        self._max_tool_iterations = max_tool_iterations
        self._tool_timeout = tool_timeout

        if mcp_client is not None:
            self._mcp_client = mcp_client
        elif mcp_servers is not None:
            from ..mcp import MCPClient

            self._mcp_client = MCPClient(mcp_servers)
        else:
            self._mcp_client = None

        self._user_tools = list(tools) if tools is not None else []
        self._tool_executor = ToolExecutor(self._user_tools, tool_timeout)
        self._tools = self._tool_executor.tools
        self._mcp_tools_synced = False

        if isinstance(telemetry, TelemetryRuntime):
            self._telemetry: TelemetryRuntime | None = telemetry
            self._owns_telemetry = False
        elif isinstance(telemetry, TelemetrySink):
            self._telemetry = TelemetryRuntime([telemetry])
            self._owns_telemetry = True
        elif isinstance(telemetry, Sequence) and telemetry and isinstance(telemetry[0], TelemetrySink):
            self._telemetry = TelemetryRuntime(telemetry)
            self._owns_telemetry = True
        else:
            self._telemetry = None
            self._owns_telemetry = False

        if retry_config is False:
            self._retry_config = RetryConfig(max_retries=0)
        elif retry_config is None and max_retries is None:
            self._retry_config = RetryConfig(max_retries=0)
        elif isinstance(retry_config, RetryConfig):
            if max_retries is not None:
                self._retry_config = replace(retry_config, max_retries=max_retries)
            else:
                self._retry_config = retry_config
        elif retry_config is True or (retry_config is None and max_retries is not None):
            mr = max_retries if max_retries is not None else 3
            self._retry_config = RetryConfig(max_retries=mr)
        else:
            raise TypeError(f"Invalid retry_config: {retry_config}")

    @property
    def retry_config(self) -> RetryConfig:
        """The retry configuration configured on this agent."""
        return self._retry_config

    @property
    def telemetry(self) -> TelemetryRuntime | None:
        """The TelemetryRuntime configured on this agent, if any."""
        return self._telemetry


    @property
    def messages(self) -> tuple[Message, ...]:
        """The retained conversation, including any generated summary message."""
        return tuple(self._messages)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The tools configured on this agent."""
        self._sync_mcp_tools()
        return self._tools

    @property
    def mcp_client(self) -> Any:
        """The MCPClient associated with this agent, if configured."""
        return self._mcp_client

    def _sync_mcp_tools(self, force: bool = False) -> None:
        """Synchronize tools from any configured MCP client into the executor."""
        if self._mcp_client is None:
            return
        if self._mcp_tools_synced and not force:
            return
        mcp_tools = self._mcp_client.get_tools()
        self._tool_executor = ToolExecutor([*self._user_tools, *mcp_tools], self._tool_timeout)
        self._tools = self._tool_executor.tools
        self._mcp_tools_synced = True

    async def _sync_mcp_tools_async(self, force: bool = False) -> None:
        """Asynchronously synchronize tools from any configured MCP client into the executor."""
        if self._mcp_client is None:
            return
        if self._mcp_tools_synced and not force:
            return
        mcp_tools = await self._mcp_client.get_tools_async()
        self._tool_executor = ToolExecutor([*self._user_tools, *mcp_tools], self._tool_timeout)
        self._tools = self._tool_executor.tools
        self._mcp_tools_synced = True

    @property
    def tool_timeout(self) -> float | None:
        """Default per-tool execution timeout in seconds, if configured."""
        return self._tool_timeout

    @property
    def summary_provider(self) -> ModelProvider:
        """The provider used for compacting and summarizing conversations."""
        return self._summary_provider

    def _generate_with_telemetry(self, request: GenerationRequest) -> GenerationResponse:
        provider_name = getattr(self._provider, "__class__", type(self._provider)).__name__
        model_name = getattr(self._provider, "model", "unknown")
        input_messages = tuple({"role": m.role, "content": m.content} for m in request.messages)
        max_retries = self._retry_config.max_retries
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            llm_ctx = create_child_trace_context()
            emit_event(
                LLMCallStartEvent(
                    trace_context=llm_ctx,
                    provider=provider_name,
                    model=model_name,
                    messages_count=len(request.messages),
                    tools_count=len(request.tools),
                    input_prompt=request.prompt,
                    input_messages=input_messages,
                    retry_attempt=attempt,
                )
            )
            t0 = time.perf_counter()
            with trace_scope(llm_ctx):
                try:
                    response = self._provider.generate(request)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    output_tool_calls = tuple(
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    )
                    token_usage = getattr(response, "usage", None)
                    emit_event(
                        LLMCallEndEvent(
                            trace_context=llm_ctx,
                            provider=provider_name,
                            model=response.model or model_name,
                            status="ok",
                            duration_ms=duration_ms,
                            token_usage=token_usage,
                            tool_calls_count=len(response.tool_calls),
                            output_text=response.text,
                            output_tool_calls=output_tool_calls,
                            retry_attempt=attempt,
                        )
                    )
                    return response
                except Exception as err:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    err_msg = str(err)
                    is_retryable = self._retry_config.is_retryable(err)
                    if attempt < max_retries and is_retryable:

                        delay = calculate_backoff_delay(attempt, self._retry_config)
                        emit_event(
                            ErrorEvent(
                                trace_context=llm_ctx,
                                error_type=type(err).__name__,
                                message=err_msg,
                                retry_count=attempt + 1,
                                outcome="retrying",
                                details={
                                    "retry_delay_seconds": delay,
                                    "max_retries": max_retries,
                                    "provider": provider_name,
                                    "model": model_name,
                                },
                            )
                        )
                        emit_event(
                            LLMCallEndEvent(
                                trace_context=llm_ctx,
                                provider=provider_name,
                                model=model_name,
                                status="error",
                                duration_ms=duration_ms,
                                error=err_msg,
                                retry_attempt=attempt,
                            )
                        )
                        self._retry_config.sleep_fn(delay)
                    else:
                        emit_event(
                            ErrorEvent(
                                trace_context=llm_ctx,
                                error_type=type(err).__name__,
                                message=err_msg,
                                retry_count=attempt,
                                outcome="exhausted" if attempt > 0 else "failed",
                                details={
                                    "max_retries": max_retries,
                                    "provider": provider_name,
                                    "model": model_name,
                                },
                            )
                        )
                        emit_event(
                            LLMCallEndEvent(
                                trace_context=llm_ctx,
                                provider=provider_name,
                                model=model_name,
                                status="error",
                                duration_ms=duration_ms,
                                error=err_msg,
                                retry_attempt=attempt,
                            )
                        )
                        raise



    def send(self, message: str) -> GenerationResponse:
        """Advance one user turn and execute any requested tool calls."""
        run_id = f"run_{secrets.token_hex(6)}"
        current_ctx = get_current_trace_context()
        root_ctx = current_ctx.child() if current_ctx is not None else TraceContext.new_root()

        with telemetry_scope(self._telemetry), trace_scope(root_ctx):
            emit_event(
                AgentRunStartEvent(
                    trace_context=root_ctx,
                    run_id=run_id,
                    user_prompt=message,
                    system_prompt=self._system_prompt,
                )
            )
            t0 = time.perf_counter()
            iterations = 0
            try:
                self._sync_mcp_tools()
                self._history.add_user(message)
                for _ in range(self._max_tool_iterations):
                    iterations += 1
                    self._compact()
                    response = self._generate_with_telemetry(
                        GenerationRequest(messages=self._messages, tools=self._tools)
                    )
                    self._history.add_assistant(response)
                    if not response.tool_calls:
                        duration_ms = (time.perf_counter() - t0) * 1000
                        emit_event(
                            AgentRunEndEvent(
                                trace_context=root_ctx,
                                run_id=run_id,
                                status="ok",
                                duration_ms=duration_ms,
                                iterations=iterations,
                                output_text=response.text,
                            )
                        )
                        return response
                    self._execute_tool_calls(response.tool_calls)

                iterations += 1
                self._compact()
                final_response = self._generate_with_telemetry(
                    GenerationRequest(messages=self._messages, tools=())
                )
                result = self._finalize_loop_expiry(final_response, response)
                duration_ms = (time.perf_counter() - t0) * 1000
                emit_event(
                    AgentRunEndEvent(
                        trace_context=root_ctx,
                        run_id=run_id,
                        status="ok",
                        duration_ms=duration_ms,
                        iterations=iterations,
                        output_text=result.text,
                    )
                )
                return result
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                err_msg = str(err)
                emit_event(
                    AgentRunEndEvent(
                        trace_context=root_ctx,
                        run_id=run_id,
                        status="error",
                        duration_ms=duration_ms,
                        iterations=max(1, iterations),
                        error=err_msg,
                    )
                )
                emit_event(
                    ErrorEvent(
                        trace_context=root_ctx,
                        error_type=type(err).__name__,
                        message=err_msg,
                    )
                )
                raise

    async def send_async(self, message: str) -> GenerationResponse:
        """Asynchronously advance one user turn and execute requested tool calls."""
        run_id = f"run_{secrets.token_hex(6)}"
        current_ctx = get_current_trace_context()
        root_ctx = current_ctx.child() if current_ctx is not None else TraceContext.new_root()

        with telemetry_scope(self._telemetry), trace_scope(root_ctx):
            emit_event(
                AgentRunStartEvent(
                    trace_context=root_ctx,
                    run_id=run_id,
                    user_prompt=message,
                    system_prompt=self._system_prompt,
                )
            )
            t0 = time.perf_counter()
            iterations = 0
            try:
                await self._sync_mcp_tools_async()
                self._history.add_user(message)
                for _ in range(self._max_tool_iterations):
                    iterations += 1
                    await self._compact_async()
                    response = await asyncio.to_thread(
                        self._generate_with_telemetry,
                        GenerationRequest(messages=self._messages, tools=self._tools),
                    )
                    self._history.add_assistant(response)
                    if not response.tool_calls:
                        duration_ms = (time.perf_counter() - t0) * 1000
                        emit_event(
                            AgentRunEndEvent(
                                trace_context=root_ctx,
                                run_id=run_id,
                                status="ok",
                                duration_ms=duration_ms,
                                iterations=iterations,
                                output_text=response.text,
                            )
                        )
                        return response
                    await self._execute_tool_calls_async(response.tool_calls)

                iterations += 1
                await self._compact_async()
                final_response = await asyncio.to_thread(
                    self._generate_with_telemetry,
                    GenerationRequest(messages=self._messages, tools=()),
                )
                result = self._finalize_loop_expiry(final_response, response)
                duration_ms = (time.perf_counter() - t0) * 1000
                emit_event(
                    AgentRunEndEvent(
                        trace_context=root_ctx,
                        run_id=run_id,
                        status="ok",
                        duration_ms=duration_ms,
                        iterations=iterations,
                        output_text=result.text,
                    )
                )
                return result
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                err_msg = str(err)
                emit_event(
                    AgentRunEndEvent(
                        trace_context=root_ctx,
                        run_id=run_id,
                        status="error",
                        duration_ms=duration_ms,
                        iterations=max(1, iterations),
                        error=err_msg,
                    )
                )
                emit_event(
                    ErrorEvent(
                        trace_context=root_ctx,
                        error_type=type(err).__name__,
                        message=err_msg,
                    )
                )
                raise

    def close(self) -> None:
        """Close any associated MCP client connections and flush telemetry."""
        if self._mcp_client is not None:
            self._mcp_client.close()
            self._mcp_tools_synced = False
        if self._telemetry is not None:
            self._telemetry.flush()
            if self._owns_telemetry:
                self._telemetry.close()

    async def close_async(self) -> None:
        """Asynchronously close any associated MCP client connections and flush telemetry."""
        if self._mcp_client is not None:
            await self._mcp_client.close_async()
            self._mcp_tools_synced = False
        if self._telemetry is not None:
            self._telemetry.flush()
            if self._owns_telemetry:
                self._telemetry.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()

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
