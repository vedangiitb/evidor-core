"""Langfuse adapter converting Evidor events into Langfuse traces and observations.

Provides production observability, prompt management, and LLM evaluation
via the Langfuse platform.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from ..events import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    MCPCallEndEvent,
    MCPCallStartEvent,
    TelemetryEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ..sink import TelemetrySink

try:
    from langfuse import Langfuse

    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    Langfuse = Any  # type: ignore[assignment, misc]


class LangfuseSink(TelemetrySink):
    """Translates Evidor events into Langfuse traces, generations, and tool spans."""

    def __init__(
        self,
        client: Any = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not HAS_LANGFUSE:
            raise ImportError(
                "The 'langfuse' package is required for the Langfuse adapter. "
                "Install it with: pip install 'evidor[langfuse]'"
            )

        if client is not None:
            self._client = client
        else:
            client_kwargs = dict(kwargs)
            if public_key:
                client_kwargs["public_key"] = public_key
            if secret_key:
                client_kwargs["secret_key"] = secret_key
            if host:
                client_kwargs["host"] = host
            self._client = Langfuse(**client_kwargs)

        self._active_observations: dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        """Process a batch of emitted events and update Langfuse observations."""
        for event in events:
            self._handle_event(event)

    def _build_trace_context(self, event: TelemetryEvent) -> dict[str, str]:
        ctx: dict[str, str] = {"trace_id": event.trace_context.trace_id}
        if event.trace_context.parent_span_id:
            ctx["parent_span_id"] = event.trace_context.parent_span_id
        return ctx

    def _handle_event(self, event: TelemetryEvent) -> None:
        if isinstance(event, AgentRunStartEvent):
            self._start_agent_run(event)
        elif isinstance(event, AgentRunEndEvent):
            self._end_agent_run(event)
        elif isinstance(event, LLMCallStartEvent):
            self._start_llm_call(event)
        elif isinstance(event, LLMCallEndEvent):
            self._end_llm_call(event)
        elif isinstance(event, ToolCallStartEvent):
            self._start_tool_call(event)
        elif isinstance(event, ToolCallEndEvent):
            self._end_tool_call(event)
        elif isinstance(event, MCPCallStartEvent):
            self._start_mcp_call(event)
        elif isinstance(event, MCPCallEndEvent):
            self._end_mcp_call(event)
        elif isinstance(event, ErrorEvent):
            self._record_error(event)

    def _start_agent_run(self, event: AgentRunStartEvent) -> None:
        trace_ctx = self._build_trace_context(event)
        metadata = {"run_id": event.run_id, **event.attributes}

        if hasattr(self._client, "start_observation"):
            obs = self._client.start_observation(
                trace_context=trace_ctx,
                name="agent.run",
                as_type="agent",
                input=event.user_prompt or None,
                metadata=metadata,
            )
        elif hasattr(self._client, "trace"):
            obs = self._client.trace(
                id=event.trace_context.trace_id,
                name="agent.run",
                input=event.user_prompt or None,
                metadata=metadata,
            )
        else:
            obs = None

        with self._lock:
            self._active_observations[event.trace_context.span_id] = obs

    def _end_agent_run(self, event: AgentRunEndEvent) -> None:
        with self._lock:
            obs = self._active_observations.pop(event.trace_context.span_id, None)

        if obs is None:
            return

        level = "ERROR" if event.status == "error" else "DEFAULT"
        if hasattr(obs, "update"):
            obs.update(
                output=event.output_text or None,
                status_message=event.error,
                level=level,
            )
        if hasattr(obs, "end"):
            obs.end()

    def _start_llm_call(self, event: LLMCallStartEvent) -> None:
        trace_ctx = self._build_trace_context(event)
        input_data = event.input_prompt or (list(event.input_messages) if event.input_messages else None)
        metadata = {"provider": event.provider, **event.attributes}
        name = f"llm.{event.model}" if event.model else "llm.call"

        if hasattr(self._client, "start_observation"):
            obs = self._client.start_observation(
                trace_context=trace_ctx,
                name=name,
                as_type="generation",
                model=event.model,
                input=input_data,
                metadata=metadata,
            )
        elif hasattr(self._client, "generation"):
            obs = self._client.generation(
                id=event.trace_context.span_id,
                trace_id=event.trace_context.trace_id,
                parent_observation_id=event.trace_context.parent_span_id,
                name=name,
                model=event.model,
                input=input_data,
                metadata=metadata,
            )
        else:
            obs = None

        with self._lock:
            self._active_observations[event.trace_context.span_id] = obs

    def _end_llm_call(self, event: LLMCallEndEvent) -> None:
        with self._lock:
            obs = self._active_observations.pop(event.trace_context.span_id, None)

        if obs is None:
            return

        usage_details = None
        if event.token_usage:
            usage_details = {
                "input": event.token_usage.prompt_tokens,
                "output": event.token_usage.completion_tokens,
                "total": event.token_usage.total_tokens,
            }

        output_data = event.output_text or (list(event.output_tool_calls) if event.output_tool_calls else None)
        level = "ERROR" if event.status == "error" else "DEFAULT"

        if hasattr(obs, "update"):
            obs.update(
                output=output_data,
                usage_details=usage_details,
                status_message=event.error,
                level=level,
            )
        if hasattr(obs, "end"):
            obs.end()

    def _start_tool_call(self, event: ToolCallStartEvent) -> None:
        trace_ctx = self._build_trace_context(event)
        metadata = {"tool_call_id": event.tool_call_id, **event.attributes}
        name = f"tool.{event.tool_name}"

        if hasattr(self._client, "start_observation"):
            obs = self._client.start_observation(
                trace_context=trace_ctx,
                name=name,
                as_type="tool",
                input=event.arguments,
                metadata=metadata,
            )
        elif hasattr(self._client, "span"):
            obs = self._client.span(
                id=event.trace_context.span_id,
                trace_id=event.trace_context.trace_id,
                parent_observation_id=event.trace_context.parent_span_id,
                name=name,
                input=event.arguments,
                metadata=metadata,
            )
        else:
            obs = None

        with self._lock:
            self._active_observations[event.trace_context.span_id] = obs

    def _end_tool_call(self, event: ToolCallEndEvent) -> None:
        with self._lock:
            obs = self._active_observations.pop(event.trace_context.span_id, None)

        if obs is None:
            return

        level = "ERROR" if event.status == "error" else "DEFAULT"
        if hasattr(obs, "update"):
            obs.update(
                output=event.result,
                status_message=event.error,
                level=level,
            )
        if hasattr(obs, "end"):
            obs.end()

    def _start_mcp_call(self, event: MCPCallStartEvent) -> None:
        trace_ctx = self._build_trace_context(event)
        metadata = {"server_name": event.server_name, **event.attributes}
        name = f"mcp.{event.server_name}.{event.tool_name}"

        if hasattr(self._client, "start_observation"):
            obs = self._client.start_observation(
                trace_context=trace_ctx,
                name=name,
                as_type="tool",
                input=event.arguments,
                metadata=metadata,
            )
        elif hasattr(self._client, "span"):
            obs = self._client.span(
                id=event.trace_context.span_id,
                trace_id=event.trace_context.trace_id,
                parent_observation_id=event.trace_context.parent_span_id,
                name=name,
                input=event.arguments,
                metadata=metadata,
            )
        else:
            obs = None

        with self._lock:
            self._active_observations[event.trace_context.span_id] = obs

    def _end_mcp_call(self, event: MCPCallEndEvent) -> None:
        with self._lock:
            obs = self._active_observations.pop(event.trace_context.span_id, None)

        if obs is None:
            return

        level = "ERROR" if event.status == "error" else "DEFAULT"
        if hasattr(obs, "update"):
            obs.update(
                status_message=event.error,
                level=level,
            )
        if hasattr(obs, "end"):
            obs.end()

    def _record_error(self, event: ErrorEvent) -> None:
        trace_ctx = self._build_trace_context(event)
        if event.trace_context.span_id:
            trace_ctx["parent_span_id"] = event.trace_context.span_id

        if hasattr(self._client, "create_event"):
            self._client.create_event(
                trace_context=trace_ctx,
                name="error",
                level="ERROR",
                status_message=event.message,
                metadata={"error_type": event.error_type, **event.details},
            )

    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if hasattr(self._client, "flush"):
            try:
                self._client.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Flush and shut down Langfuse background client threads."""
        self.flush()
        if hasattr(self._client, "shutdown"):
            try:
                self._client.shutdown()
            except Exception:
                pass
