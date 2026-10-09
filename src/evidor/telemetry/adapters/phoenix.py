"""Arize Phoenix adapter exporting Evidor traces with OpenInference conventions.

Sends distributed trace spans to Arize Phoenix (local or cloud) formatted with
OpenInference semantic conventions for AI Agent and LLM visualization.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..events import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    MCPCallEndEvent,
    MCPCallStartEvent,
    TelemetryEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .opentelemetry import OpenTelemetrySink

try:
    import phoenix.otel as potel

    HAS_PHOENIX = True
except ImportError:
    HAS_PHOENIX = False
    potel = None  # type: ignore[assignment]


class PhoenixSink(OpenTelemetrySink):
    """Arize Phoenix telemetry sink formatting events according to OpenInference standards."""

    def __init__(
        self,
        endpoint: str | None = None,
        project_name: str = "evidor-agent",
        tracer_provider: Any = None,
        tracer_name: str = "evidor",
        **register_kwargs: Any,
    ) -> None:
        if not HAS_PHOENIX:
            raise ImportError(
                "The 'arize-phoenix-otel' and 'openinference-semantic-conventions' packages are required "
                "for the Phoenix adapter. Install them with: pip install 'evidor[phoenix]'"
            )

        if tracer_provider is None:
            # Register Arize Phoenix tracer provider
            tracer_provider = potel.register(
                endpoint=endpoint or "http://localhost:6006/v1/traces",
                project_name=project_name,
                set_global_tracer_provider=False,
                auto_instrument=False,
                verbose=False,
                **register_kwargs,
            )

        super().__init__(
            tracer_provider=tracer_provider,
            tracer_name=tracer_name,
        )

    def _start_span(self, event: TelemetryEvent) -> None:
        super()._start_span(event)
        with self._lock:
            span = self._active_spans.get(event.trace_context.span_id)

        if span is None:
            return

        # OpenInference specific enrichments
        if isinstance(event, AgentRunStartEvent):
            if event.user_prompt:
                span.set_attribute("input.value", event.user_prompt)

        elif isinstance(event, LLMCallStartEvent):
            span.set_attribute("llm.model_name", event.model)
            if event.input_prompt:
                span.set_attribute("input.value", event.input_prompt)
            elif event.input_messages:
                span.set_attribute("input.value", json.dumps(list(event.input_messages)))

        elif isinstance(event, ToolCallStartEvent):
            span.set_attribute("tool.name", event.tool_name)
            args_str = json.dumps(event.arguments) if event.arguments else "{}"
            span.set_attribute("tool.parameters", args_str)
            span.set_attribute("input.value", args_str)

        elif isinstance(event, MCPCallStartEvent):
            span.set_attribute("tool.name", f"{event.server_name}.{event.tool_name}")
            args_str = json.dumps(event.arguments) if event.arguments else "{}"
            span.set_attribute("tool.parameters", args_str)
            span.set_attribute("input.value", args_str)

    def _end_span(self, event: TelemetryEvent) -> None:
        with self._lock:
            span = self._active_spans.get(event.trace_context.span_id)

        if span is not None:
            # Set OpenInference output and token attributes before span closes
            if isinstance(event, AgentRunEndEvent):
                if event.output_text:
                    span.set_attribute("output.value", event.output_text)

            elif isinstance(event, LLMCallEndEvent):
                if event.output_text:
                    span.set_attribute("output.value", event.output_text)
                if event.token_usage:
                    if event.token_usage.prompt_tokens > 0:
                        span.set_attribute("llm.token_count.prompt", event.token_usage.prompt_tokens)
                    if event.token_usage.completion_tokens > 0:
                        span.set_attribute("llm.token_count.completion", event.token_usage.completion_tokens)
                    if event.token_usage.total_tokens > 0:
                        span.set_attribute("llm.token_count.total", event.token_usage.total_tokens)

            elif isinstance(event, ToolCallEndEvent):
                if event.result is not None:
                    span.set_attribute("output.value", str(event.result))

            elif isinstance(event, MCPCallEndEvent):
                span.set_attribute("output.value", str(event.status))

        super()._end_span(event)
