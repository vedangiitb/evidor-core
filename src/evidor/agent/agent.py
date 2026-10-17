"""The public conversational agent and session facade."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import inspect
import secrets
import time
from typing import Any

from ..models import GenerationRequest, GenerationResponse, Message, ToolCall
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
from ..utils import run_async_in_sync
from .providers.base import ModelProvider


class Agent:
    """A conversational session coordinating a provider, context state, tools, and telemetry.

    Context management distinguishes between provider-managed server-side chains and stateless replay:
    - OpenAI: Provider-managed server-side state chained via `previous_response_id` (Responses API).
    - Gemini: Provider-managed server-side state chained via `previous_interaction_id` (Interactions API).
    - Anthropic: Stateless client-replayed history (`stateless_replay`), sending multi-turn messages
      and leveraging Anthropic's ephemeral prompt caching (`cache_control`) for prefix cost/latency savings.
    """

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
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        store: bool | None = None,
        compact_threshold: int | None = None,
        telemetry: TelemetryRuntime | TelemetrySink | Sequence[TelemetrySink] | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._last_response_id = previous_response_id
        self._conversation_id = conversation_id
        self._store = store
        self._compact_threshold = compact_threshold or getattr(provider, "compact_threshold", None)
        if (
            compact_threshold is not None
            and hasattr(provider, "_compact_threshold")
            and getattr(provider, "_compact_threshold") is None
        ):
            provider._compact_threshold = compact_threshold

        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

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

    @property
    def provider(self) -> ModelProvider:
        """The underlying model provider."""
        return self._provider

    @property
    def strategy(self) -> str:
        """The context management strategy employed by this provider:
        - 'response_chain': OpenAI Responses API server-side chaining
        - 'interaction_chain': Gemini Interactions API server-side chaining
        - 'stateless_replay': Replays message history (used by Anthropic Claude and stateless models)
        """
        return getattr(self._provider, "context_strategy", "stateless_replay")

    @property
    def last_response_id(self) -> str | None:
        """The identifier of the last response or interaction in this conversation."""
        return self._last_response_id

    @property
    def conversation_id(self) -> str | None:
        """The conversation identifier, if attached to an external conversation resource."""
        return self._conversation_id

    @property
    def system_prompt(self) -> str | None:
        """The optional system instructions configured for this agent."""
        return self._system_prompt

    @property
    def compact_threshold(self) -> int | None:
        """The configured token threshold for provider-native compaction, if any."""
        return self._compact_threshold

    @property
    def telemetry(self) -> TelemetryRuntime | None:
        """The TelemetryRuntime configured on this agent, if any."""
        return self._telemetry

    @property
    def messages(self) -> tuple[Message, ...]:
        """The recorded conversation history."""
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

    @property
    def tool_timeout(self) -> float | None:
        """Default per-tool execution timeout in seconds, if configured."""
        return self._tool_timeout

    def clear_history(self) -> None:
        """Clear conversation history and reset provider state identifiers."""
        self._last_response_id = None
        self._messages = []
        if self._system_prompt:
            self._messages.append(Message(role="system", content=self._system_prompt))

    def reset(self) -> None:
        """Alias for clear_history."""
        self.clear_history()

    def _sync_mcp_tools(self, force: bool = False) -> None:
        """Synchronize tools from any configured MCP client into the executor."""
        if self._mcp_client is None:
            return
        if self._mcp_tools_synced and not force:
            return
        run_async_in_sync(
            lambda: self._sync_mcp_tools_async(force=force),
            error_message=(
                "Cannot synchronize MCP tools synchronously from a running event loop. "
                "Access agent.tools outside an event loop or run an async agent turn first."
            ),
        )

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

    def _emit_llm_start(
        self,
        llm_ctx: TraceContext,
        provider_name: str,
        model_name: str,
        request: GenerationRequest,
        call_type: str = "generation",
    ) -> None:
        input_messages = tuple({"role": m.role, "content": m.content} for m in request.messages)
        emit_event(
            LLMCallStartEvent(
                trace_context=llm_ctx,
                provider=provider_name,
                model=model_name,
                call_type=call_type,
                messages_count=len(request.messages),
                tools_count=len(request.tools),
                input_prompt=request.prompt,
                input_messages=input_messages,
            )
        )

    def _emit_llm_success(
        self,
        llm_ctx: TraceContext,
        provider_name: str,
        model_name: str,
        response: GenerationResponse,
        duration_ms: float,
        call_type: str = "generation",
    ) -> None:
        output_tool_calls = tuple(
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        )
        token_usage = getattr(response, "usage", None)
        thinking_text = getattr(response, "thinking", None)
        emit_event(
            LLMCallEndEvent(
                trace_context=llm_ctx,
                provider=provider_name,
                model=response.model or model_name,
                call_type=call_type,
                status="ok",
                duration_ms=duration_ms,
                token_usage=token_usage,
                tool_calls_count=len(response.tool_calls),
                output_text=response.text,
                output_tool_calls=output_tool_calls,
                thinking_text=thinking_text,
            )
        )

    def _emit_llm_error(
        self,
        llm_ctx: TraceContext,
        provider_name: str,
        model_name: str,
        err: Exception,
        duration_ms: float,
        call_type: str = "generation",
    ) -> None:
        err_msg = str(err)
        emit_event(
            ErrorEvent(
                trace_context=llm_ctx,
                error_type=type(err).__name__,
                message=err_msg,
                outcome="failed",
                details={
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
                call_type=call_type,
                status="error",
                duration_ms=duration_ms,
                error=err_msg,
            )
        )

    async def _generate_with_telemetry_async(
        self,
        request: GenerationRequest,
        provider: ModelProvider | None = None,
        call_type: str = "generation",
    ) -> GenerationResponse:
        target_provider = provider if provider is not None else self._provider
        provider_name = getattr(target_provider, "__class__", type(target_provider)).__name__
        model_name = getattr(target_provider, "model", "unknown")

        llm_ctx = create_child_trace_context()
        self._emit_llm_start(llm_ctx, provider_name, model_name, request, call_type=call_type)
        t0 = time.perf_counter()
        with trace_scope(llm_ctx):
            try:
                gen_async = getattr(target_provider, "generate_async", None)
                if callable(gen_async) and inspect.iscoroutinefunction(gen_async):
                    response = await gen_async(request)
                else:
                    response = await asyncio.to_thread(target_provider.generate, request)
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                self._emit_llm_error(
                    llm_ctx,
                    provider_name,
                    model_name,
                    err,
                    duration_ms,
                    call_type=call_type,
                )
                raise

            duration_ms = (time.perf_counter() - t0) * 1000
            self._emit_llm_success(
                llm_ctx, provider_name, model_name, response, duration_ms, call_type=call_type
            )
            return response

    def send(self, message: str) -> GenerationResponse:
        """Advance one user turn synchronously."""
        return run_async_in_sync(
            lambda: self.send_async(message),
            error_message="Agent.send() cannot be called from a running event loop. Use 'await agent.send_async()' instead.",
        )

    async def send_async(self, message: str) -> GenerationResponse:
        """Asynchronously advance one user turn, coordinating provider-native context and tool calls."""
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
                user_msg = Message(role="user", content=message)
                self._messages.append(user_msg)

                strategy = self.strategy
                is_stateful_chain = strategy in ("response_chain", "interaction_chain")
                current_turn_messages = [user_msg]
                response = None

                for _ in range(self._max_tool_iterations):
                    iterations += 1
                    if is_stateful_chain:
                        req = GenerationRequest(
                            messages=current_turn_messages,
                            tools=self._tools,
                            previous_response_id=self._last_response_id,
                            conversation_id=self._conversation_id,
                            store=self._store,
                            system=self._system_prompt,
                        )
                    else:
                        req = GenerationRequest(
                            messages=self._messages,
                            tools=self._tools,
                            system=self._system_prompt,
                        )

                    response = await self._generate_with_telemetry_async(req)
                    if response.id:
                        self._last_response_id = response.id

                    asst_content = response.text
                    if (
                        (not asst_content or not asst_content.strip())
                        and not response.tool_calls
                        and not response.thought_blocks
                        and not getattr(response, "compaction", None)
                    ):
                        asst_content = "Done."
                    self._messages.append(
                        Message(
                            role="assistant",
                            content=asst_content,
                            tool_calls=response.tool_calls,
                            thinking=response.thinking,
                            thought_blocks=response.thought_blocks,
                            compaction=getattr(response, "compaction", None),
                        )
                    )

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

                    # Execute tool calls asynchronously (concurrently if multiple)
                    if len(response.tool_calls) == 1:
                        call = response.tool_calls[0]
                        result = await self._tool_executor.execute_async(call)
                        tool_results = [
                            Message(
                                role="tool",
                                content=result,
                                tool_call_id=call.id,
                                name=call.name,
                            )
                        ]
                    else:
                        async def _exec_tool_call(call: ToolCall) -> Message:
                            res = await self._tool_executor.execute_async(call)
                            return Message(
                                role="tool",
                                content=res,
                                tool_call_id=call.id,
                                name=call.name,
                            )

                        tool_results = list(
                            await asyncio.gather(*(_exec_tool_call(c) for c in response.tool_calls))
                        )

                    for t_msg in tool_results:
                        self._messages.append(t_msg)
                    current_turn_messages = tool_results

                iterations += 1
                if is_stateful_chain:
                    final_req = GenerationRequest(
                        messages=current_turn_messages,
                        tools=(),
                        previous_response_id=self._last_response_id,
                        conversation_id=self._conversation_id,
                        store=self._store,
                        system=self._system_prompt,
                    )
                else:
                    final_req = GenerationRequest(
                        messages=self._messages,
                        tools=(),
                        system=self._system_prompt,
                    )
                final_response = await self._generate_with_telemetry_async(final_req)
                if final_response.id:
                    self._last_response_id = final_response.id
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
            run_async_in_sync(
                self.close_async,
                error_message=(
                    "Agent.close() cannot be called from a running event loop. "
                    "Use 'await agent.close_async()' or 'async with agent:' instead."
                ),
            )
        else:
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

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()

    def _finalize_loop_expiry(
        self, final_response: GenerationResponse, last_response: GenerationResponse | None
    ) -> GenerationResponse:
        final_text = (final_response.text or "").strip()
        if not final_text and last_response:
            final_text = (last_response.text or "").strip()
        if not final_text:
            final_text = f"Reached maximum tool iterations ({self._max_tool_iterations}) without a final response."
        self._messages.append(Message(role="assistant", content=final_text))
        return GenerationResponse(
            id=final_response.id or (last_response.id if last_response else None),
            text=final_text,
            model=final_response.model or (last_response.model if last_response else "unknown"),
        )
