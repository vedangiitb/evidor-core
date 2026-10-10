"""Langfuse adapter converting Evidor events into Langfuse traces and observations."""

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
    redact_event,
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

    def __init__(self, client: Any = None, capture_content: bool = True, **kwargs: Any) -> None:
        if not HAS_LANGFUSE:
            raise ImportError(
                "The 'langfuse' package is required for the Langfuse adapter. "
                "Install it with: pip install 'evidor[langfuse]'"
            )

        self._client = client if client is not None else Langfuse(**kwargs)
        self._capture_content = capture_content
        self._active_observations: dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        """Process a batch of emitted events."""
        for event in events:
            ev = event if self._capture_content else redact_event(event)
            if isinstance(ev, (AgentRunStartEvent, LLMCallStartEvent, ToolCallStartEvent, MCPCallStartEvent)):
                self._start_observation(ev)
            elif isinstance(ev, (AgentRunEndEvent, LLMCallEndEvent, ToolCallEndEvent, MCPCallEndEvent)):
                self._end_observation(ev)
            elif isinstance(ev, ErrorEvent):
                self._record_error(ev)

    def _trace_context(self, event: TelemetryEvent) -> dict[str, str]:
        ctx = {"trace_id": event.trace_context.trace_id}
        if event.trace_context.parent_span_id:
            ctx["parent_span_id"] = event.trace_context.parent_span_id
        return ctx

    def _start_observation(self, event: TelemetryEvent) -> None:
        params: dict[str, Any] = {
            "trace_context": self._trace_context(event),
            "metadata": dict(event.attributes),
        }

        if isinstance(event, AgentRunStartEvent):
            params.update(name="agent.run", as_type="agent", input=event.user_prompt or None)
            params["metadata"]["run_id"] = event.run_id
        elif isinstance(event, LLMCallStartEvent):
            inp = event.input_prompt or (list(event.input_messages) if event.input_messages else None)
            params.update(name=f"llm.{event.model}" if event.model else "llm.call", as_type="generation", model=event.model, input=inp)
            params["metadata"]["provider"] = event.provider
        elif isinstance(event, ToolCallStartEvent):
            params.update(name=f"tool.{event.tool_name}", as_type="tool", input=event.arguments)
            params["metadata"]["tool_call_id"] = event.tool_call_id
        elif isinstance(event, MCPCallStartEvent):
            params.update(name=f"mcp.{event.server_name}.{event.tool_name}", as_type="tool", input=event.arguments)
            params["metadata"]["server_name"] = event.server_name

        obs = None
        if hasattr(self._client, "start_observation"):
            obs = self._client.start_observation(**params)
        elif hasattr(self._client, "trace") and isinstance(event, AgentRunStartEvent):
            obs = self._client.trace(id=event.trace_context.trace_id, name=params["name"], input=params.get("input"), metadata=params["metadata"])

        with self._lock:
            self._active_observations[event.trace_context.span_id] = obs

    def _end_observation(self, event: TelemetryEvent) -> None:
        with self._lock:
            obs = self._active_observations.pop(event.trace_context.span_id, None)

        if not obs:
            return

        out = getattr(event, "output_text", None) or getattr(event, "result", None)
        if not out and getattr(event, "output_tool_calls", None):
            out = list(event.output_tool_calls)

        usage = None
        tok = getattr(event, "token_usage", None)
        if tok:
            usage = {"input": tok.prompt_tokens, "output": tok.completion_tokens, "total": tok.total_tokens}

        level = "ERROR" if getattr(event, "status", None) == "error" else "DEFAULT"
        err = getattr(event, "error", None)

        if hasattr(obs, "update"):
            update_kwargs: dict[str, Any] = {"level": level, "status_message": err}
            if out is not None:
                update_kwargs["output"] = out
            if usage:
                update_kwargs["usage_details"] = usage
            obs.update(**update_kwargs)
        if hasattr(obs, "end"):
            obs.end()

    def _record_error(self, event: ErrorEvent) -> None:
        if hasattr(self._client, "create_event"):
            ctx = {"trace_id": event.trace_context.trace_id}
            if event.trace_context.span_id:
                ctx["parent_span_id"] = event.trace_context.span_id
            is_retry = event.outcome == "retrying"
            self._client.create_event(
                trace_context=ctx,
                name="retry" if is_retry else "error",
                level="WARNING" if is_retry else "ERROR",
                status_message=event.message,
                metadata={
                    "error_type": event.error_type,
                    "retry_count": event.retry_count,
                    "outcome": event.outcome,
                    **event.details,
                },
            )


    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        if hasattr(self._client, "flush"):
            try:
                self._client.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Flush and shut down Langfuse background threads."""
        self.flush()
        if hasattr(self._client, "shutdown"):
            try:
                self._client.shutdown()
            except Exception:
                pass
