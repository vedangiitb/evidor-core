"""Turn execution engine coordinating provider turns, tool loops, and streaming bridges."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
import queue
import secrets
import time
from typing import Any

from evidor.models import (
    GenerationRequest,
    GenerationResponse,
    Message,
    StreamChunk,
    ToolCall,
)
from evidor.tools.core.tool_executor import ToolExecutor
from evidor.utils import BackgroundEventLoop
from .context import ConversationContext
from .telemetry_bridge import AgentTelemetryBridge
from ..providers.base import ModelProvider


class TurnExecutor:
    """Executes multi-turn conversational loops with automatic tool calling.

    Adheres to the Single Responsibility Principle by isolating the orchestration
    logic (iteration bounds, tool calling loops, streaming chunk generation) from
    state storage and high-level agent lifecycle management.
    """

    def __init__(
        self,
        provider: ModelProvider,
        context: ConversationContext,
        tool_executor: ToolExecutor,
        telemetry_bridge: AgentTelemetryBridge,
        max_tool_iterations: int = 10,
    ) -> None:
        self._provider = provider
        self._context = context
        self._tool_executor = tool_executor
        self._telemetry_bridge = telemetry_bridge
        self._max_tool_iterations = max_tool_iterations

    def update_tool_executor(self, executor: ToolExecutor) -> None:
        """Update the active ToolExecutor (e.g. after MCP tool synchronization)."""
        self._tool_executor = executor

    async def execute_tool_calls_async(self, tool_calls: Sequence[ToolCall]) -> list[Message]:
        """Execute one or more tool calls asynchronously (concurrently when multiple)."""
        if len(tool_calls) == 1:
            call = tool_calls[0]
            result = await self._tool_executor.execute_async(call)
            return [
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=call.id,
                    name=call.name,
                )
            ]

        async def _exec_single(call: ToolCall) -> Message:
            res = await self._tool_executor.execute_async(call)
            return Message(
                role="tool",
                content=res,
                tool_call_id=call.id,
                name=call.name,
            )

        return list(await asyncio.gather(*(_exec_single(c) for c in tool_calls)))

    def finalize_loop_expiry(
        self,
        final_response: GenerationResponse,
        last_response: GenerationResponse | None,
    ) -> GenerationResponse:
        """Synthesize a fallback response if the model exhausts tool iterations without concluding."""
        final_text = (final_response.text or "").strip()
        if not final_text and last_response:
            final_text = (last_response.text or "").strip()
        if not final_text:
            final_text = (
                f"Reached maximum tool iterations ({self._max_tool_iterations}) without a final response."
            )
        self._context.append_assistant_response(text=final_text)
        return GenerationResponse(
            id=final_response.id or (last_response.id if last_response else None),
            text=final_text,
            model=final_response.model or (last_response.model if last_response else "unknown"),
        )

    async def run_turn_async(self, message: str) -> GenerationResponse:
        """Asynchronously advance one complete user turn with automatic tool calling."""
        run_id = f"run_{secrets.token_hex(6)}"
        with self._telemetry_bridge.run_scope(
            run_id, message, self._context.system_prompt
        ) as root_ctx:
            t0 = time.perf_counter()
            iterations = 0
            try:
                user_msg = self._context.append_user_message(message)
                current_turn_messages: Sequence[Message] = [user_msg]
                response: GenerationResponse | None = None

                for _ in range(self._max_tool_iterations):
                    iterations += 1
                    req = self._context.build_generation_request(
                        current_turn_messages, self._tool_executor.tools
                    )

                    response = await self._telemetry_bridge.generate_with_telemetry_async(
                        self._provider, req
                    )
                    if response.id:
                        self._context.last_response_id = response.id

                    self._context.append_assistant_response(
                        text=response.text,
                        tool_calls=response.tool_calls,
                        thinking=response.thinking,
                        thought_blocks=response.thought_blocks,
                        compaction=getattr(response, "compaction", None),
                    )

                    if not response.tool_calls:
                        duration_ms = (time.perf_counter() - t0) * 1000
                        self._telemetry_bridge.emit_run_end(
                            root_ctx=root_ctx,
                            run_id=run_id,
                            status="ok",
                            duration_ms=duration_ms,
                            iterations=iterations,
                            output_text=response.text,
                        )
                        return response

                    # Execute tool calls and record results
                    tool_results = await self.execute_tool_calls_async(response.tool_calls)
                    self._context.append_tool_results(tool_results)
                    current_turn_messages = tool_results

                # Iterations exhausted: perform terminal synthesis without tools
                iterations += 1
                final_req = self._context.build_final_request(current_turn_messages)
                final_response = await self._telemetry_bridge.generate_with_telemetry_async(
                    self._provider, final_req
                )
                if final_response.id:
                    self._context.last_response_id = final_response.id

                result = self.finalize_loop_expiry(final_response, response)
                duration_ms = (time.perf_counter() - t0) * 1000
                self._telemetry_bridge.emit_run_end(
                    root_ctx=root_ctx,
                    run_id=run_id,
                    status="ok",
                    duration_ms=duration_ms,
                    iterations=iterations,
                    output_text=result.text,
                )
                return result
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                self._telemetry_bridge.emit_run_end(
                    root_ctx=root_ctx,
                    run_id=run_id,
                    status="error",
                    duration_ms=duration_ms,
                    iterations=max(1, iterations),
                    error=str(err),
                )
                self._telemetry_bridge.emit_run_error(root_ctx, err)
                raise

    async def stream_turn_async(self, message: str) -> AsyncIterator[StreamChunk]:
        """Asynchronously stream chunks across multi-turn tool execution."""
        run_id = f"run_{secrets.token_hex(6)}"
        with self._telemetry_bridge.run_scope(
            run_id, message, self._context.system_prompt
        ) as root_ctx:
            t0 = time.perf_counter()
            iterations = 0
            try:
                user_msg = self._context.append_user_message(message)
                current_turn_messages: Sequence[Message] = [user_msg]
                response: GenerationResponse | None = None

                for _ in range(self._max_tool_iterations):
                    iterations += 1
                    req = self._context.build_generation_request(
                        current_turn_messages, self._tool_executor.tools, stream=True
                    )

                    turn_stream = await self._telemetry_bridge.generate_with_telemetry_stream_async(
                        self._provider, req
                    )
                    async for chunk in turn_stream:
                        yield chunk

                    response = turn_stream.response
                    if response is None:
                        response = await turn_stream.get_final_response()

                    if response.id:
                        self._context.last_response_id = response.id

                    self._context.append_assistant_response(
                        text=response.text,
                        tool_calls=response.tool_calls,
                        thinking=response.thinking,
                        thought_blocks=response.thought_blocks,
                        compaction=getattr(response, "compaction", None),
                    )

                    if not response.tool_calls:
                        duration_ms = (time.perf_counter() - t0) * 1000
                        self._telemetry_bridge.emit_run_end(
                            root_ctx=root_ctx,
                            run_id=run_id,
                            status="ok",
                            duration_ms=duration_ms,
                            iterations=iterations,
                            output_text=response.text,
                        )
                        return

                    # Execute tool calls asynchronously
                    tool_results = await self.execute_tool_calls_async(response.tool_calls)
                    self._context.append_tool_results(tool_results)
                    current_turn_messages = tool_results

                # Iterations exhausted: stream final synthesis without tools
                iterations += 1
                final_req = self._context.build_final_request(current_turn_messages, stream=True)
                final_stream = await self._telemetry_bridge.generate_with_telemetry_stream_async(
                    self._provider, final_req
                )
                async for chunk in final_stream:
                    yield chunk

                final_response = final_stream.response
                if final_response is None:
                    final_response = await final_stream.get_final_response()
                if final_response.id:
                    self._context.last_response_id = final_response.id

                result = self.finalize_loop_expiry(final_response, response)
                duration_ms = (time.perf_counter() - t0) * 1000
                self._telemetry_bridge.emit_run_end(
                    root_ctx=root_ctx,
                    run_id=run_id,
                    status="ok",
                    duration_ms=duration_ms,
                    iterations=iterations,
                    output_text=result.text,
                )
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                self._telemetry_bridge.emit_run_end(
                    root_ctx=root_ctx,
                    run_id=run_id,
                    status="error",
                    duration_ms=duration_ms,
                    iterations=max(1, iterations),
                    error=str(err),
                )
                self._telemetry_bridge.emit_run_error(root_ctx, err)
                raise


def sync_stream_bridge(
    bg_loop: BackgroundEventLoop,
    async_stream_coro_fn: Callable[[], Any],
) -> Iterator[StreamChunk]:
    """Bridge an asynchronous stream into a synchronous generator via an in-memory queue."""
    q: queue.Queue[Any] = queue.Queue(maxsize=100)
    sentinel = object()

    async def _producer() -> None:
        try:
            async_stream = await async_stream_coro_fn()
            async for chunk in async_stream:
                q.put(chunk)
            q.put(sentinel)
        except Exception as exc:
            q.put(exc)

    running_loop = bg_loop._ensure_running()
    asyncio.run_coroutine_threadsafe(_producer(), running_loop)

    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item

