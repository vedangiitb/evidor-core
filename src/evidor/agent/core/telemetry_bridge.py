"""Telemetry bridge coordinating trace contexts and metrics for agent runs and LLM calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import contextmanager
import inspect
import time
from typing import Any, Iterator

from evidor.models import (
    AsyncStreamResponse,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    StreamResponse,
)
from evidor.telemetry import (
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
from ..providers.base import ModelProvider


class AgentTelemetryBridge:
    """Manages telemetry emission for agent runs and provider generation calls.

    Adheres to the Single Responsibility Principle by decoupling observability
    lifecycle from agent conversational logic.
    """

    def __init__(
        self,
        telemetry: TelemetryRuntime | TelemetrySink | Sequence[TelemetrySink] | None = None,
    ) -> None:
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
    def telemetry(self) -> TelemetryRuntime | None:
        """The active TelemetryRuntime instance, if configured."""
        return self._telemetry

    @property
    def owns_telemetry(self) -> bool:
        """Whether this bridge created and owns the TelemetryRuntime lifecycle."""
        return self._owns_telemetry

    @owns_telemetry.setter
    def owns_telemetry(self, value: bool) -> None:
        self._owns_telemetry = value

    def emit_llm_start(
        self,
        llm_ctx: TraceContext,
        provider_name: str,
        model_name: str,
        request: GenerationRequest,
        call_type: str = "generation",
        is_streaming: bool = False,
    ) -> None:
        """Emit an LLMCallStartEvent capturing input messages, prompt, and tool definitions."""
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
                is_streaming=is_streaming,
            )
        )

    def emit_llm_success(
        self,
        llm_ctx: TraceContext,
        provider_name: str,
        model_name: str,
        response: GenerationResponse,
        duration_ms: float,
        call_type: str = "generation",
        time_to_first_token_ms: float | None = None,
        is_streaming: bool = False,
    ) -> None:
        """Emit an LLMCallEndEvent with status='ok', duration, and token usage."""
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
                time_to_first_token_ms=time_to_first_token_ms,
                is_streaming=is_streaming,
            )
        )

    def emit_llm_error(
        self,
        llm_ctx: TraceContext,
        provider_name: str,
        model_name: str,
        err: Exception,
        duration_ms: float,
        call_type: str = "generation",
    ) -> None:
        """Emit ErrorEvent and failed LLMCallEndEvent on provider failure."""
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

    async def generate_with_telemetry_async(
        self,
        provider: ModelProvider,
        request: GenerationRequest,
        call_type: str = "generation",
    ) -> GenerationResponse:
        """Execute a unary generation against the provider wrapped in trace spans."""
        provider_name = getattr(provider, "__class__", type(provider)).__name__
        model_name = getattr(provider, "model", "unknown")

        llm_ctx = create_child_trace_context()
        self.emit_llm_start(llm_ctx, provider_name, model_name, request, call_type=call_type)
        t0 = time.perf_counter()
        with trace_scope(llm_ctx):
            try:
                gen_async = getattr(provider, "generate_async", None)
                if callable(gen_async) and inspect.iscoroutinefunction(gen_async):
                    response = await gen_async(request)
                else:
                    response = await asyncio.to_thread(provider.generate, request)
            except Exception as err:
                duration_ms = (time.perf_counter() - t0) * 1000
                self.emit_llm_error(
                    llm_ctx,
                    provider_name,
                    model_name,
                    err,
                    duration_ms,
                    call_type=call_type,
                )
                raise

            duration_ms = (time.perf_counter() - t0) * 1000
            self.emit_llm_success(
                llm_ctx, provider_name, model_name, response, duration_ms, call_type=call_type
            )
            return response

    async def generate_with_telemetry_stream_async(
        self,
        provider: ModelProvider,
        request: GenerationRequest,
        call_type: str = "generation",
    ) -> AsyncStreamResponse:
        """Execute a streaming generation wrapped with TTFT metrics and trace spans."""
        provider_name = getattr(provider, "__class__", type(provider)).__name__
        model_name = getattr(provider, "model", "unknown")

        llm_ctx = create_child_trace_context()
        self.emit_llm_start(
            llm_ctx, provider_name, model_name, request, call_type=call_type, is_streaming=True
        )
        t0 = time.perf_counter()

        gen_async = getattr(provider, "generate_async", None)
        if callable(gen_async) and inspect.iscoroutinefunction(gen_async):
            raw_stream_resp = await gen_async(request, stream=True)
        else:
            raw_stream_resp = await asyncio.to_thread(provider.generate, request, stream=True)

        ttft_recorded: list[float] = []

        def _on_first_token(ttft_ms: float) -> None:
            ttft_recorded.append(ttft_ms)

        def _on_complete(final_resp: GenerationResponse, ttft_ms: float | None) -> None:
            duration_ms = (time.perf_counter() - t0) * 1000
            first_token = ttft_recorded[0] if ttft_recorded else ttft_ms
            with trace_scope(llm_ctx):
                self.emit_llm_success(
                    llm_ctx,
                    provider_name,
                    model_name,
                    final_resp,
                    duration_ms,
                    call_type=call_type,
                    time_to_first_token_ms=first_token,
                    is_streaming=True,
                )

        def _on_error(err: Exception) -> None:
            duration_ms = (time.perf_counter() - t0) * 1000
            with trace_scope(llm_ctx):
                self.emit_llm_error(
                    llm_ctx,
                    provider_name,
                    model_name,
                    err,
                    duration_ms,
                    call_type=call_type,
                )

        if isinstance(raw_stream_resp, AsyncStreamResponse):
            raw_stream_resp._on_first_token = _on_first_token
            raw_stream_resp._on_complete = _on_complete
            raw_stream_resp._on_error = _on_error
            return raw_stream_resp
        elif isinstance(raw_stream_resp, StreamResponse):
            async def _bridge_iter() -> AsyncIterator[StreamChunk]:
                for c in raw_stream_resp:
                    yield c

            return AsyncStreamResponse(
                _bridge_iter(),
                on_first_token=_on_first_token,
                on_complete=_on_complete,
                on_error=_on_error,
            )
        else:
            async def _single_iter() -> AsyncIterator[StreamChunk]:
                yield StreamChunk(
                    text=raw_stream_resp.text,
                    is_finished=True,
                    response=raw_stream_resp,
                )

            return AsyncStreamResponse(
                _single_iter(),
                on_first_token=_on_first_token,
                on_complete=_on_complete,
                on_error=_on_error,
            )

    @contextmanager
    def run_scope(
        self,
        run_id: str,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> Iterator[TraceContext]:
        """Establish the root trace context and emit AgentRunStartEvent."""
        current_ctx = get_current_trace_context()
        root_ctx = current_ctx.child() if current_ctx is not None else TraceContext.new_root()

        with telemetry_scope(self._telemetry), trace_scope(root_ctx):
            emit_event(
                AgentRunStartEvent(
                    trace_context=root_ctx,
                    run_id=run_id,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            yield root_ctx

    def emit_run_end(
        self,
        root_ctx: TraceContext,
        run_id: str,
        status: str,
        duration_ms: float,
        iterations: int,
        output_text: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit an AgentRunEndEvent marking agent completion."""
        emit_event(
            AgentRunEndEvent(
                trace_context=root_ctx,
                run_id=run_id,
                status=status,
                duration_ms=duration_ms,
                iterations=iterations,
                output_text=output_text,
                error=error,
            )
        )

    def emit_run_error(
        self,
        root_ctx: TraceContext,
        err: Exception,
    ) -> None:
        """Emit an ErrorEvent for an unhandled exception during an agent run."""
        emit_event(
            ErrorEvent(
                trace_context=root_ctx,
                error_type=type(err).__name__,
                message=str(err),
            )
        )

    def flush_and_close(self) -> None:
        """Flush and shut down owned telemetry resources."""
        if self._telemetry is not None:
            try:
                self._telemetry.flush()
            except Exception:
                pass
            if self._owns_telemetry:
                try:
                    self._telemetry.close()
                except Exception:
                    pass
                self._owns_telemetry = False

