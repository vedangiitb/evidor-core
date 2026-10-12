"""Arize Phoenix adapter exporting Evidor traces with OpenInference conventions.

Sends distributed trace spans to Arize Phoenix (local or cloud) formatted with
OpenInference semantic conventions for AI Agent and LLM visualization.
"""

from __future__ import annotations

import json
import warnings
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
        capture_content: bool = True,
        **register_kwargs: Any,
    ) -> None:
        if not HAS_PHOENIX:
            raise ImportError(
                "The 'arize-phoenix-otel' and 'openinference-semantic-conventions' packages are required "
                "for the Phoenix adapter. Install them with: pip install 'evidor[phoenix]'"
            )

        if tracer_provider is None:
            tracer_provider = self._register_tracer_provider(
                endpoint=endpoint or "http://localhost:6006/v1/traces",
                project_name=project_name,
                register_kwargs=register_kwargs,
            )

        super().__init__(
            tracer_provider=tracer_provider,
            tracer_name=tracer_name,
            capture_content=capture_content,
        )

    @staticmethod
    def _register_tracer_provider(
        *,
        endpoint: str,
        project_name: str,
        register_kwargs: dict[str, Any],
    ) -> Any:
        """Register Phoenix tracing, tolerating incompatible OTEL private APIs.

        Some Phoenix releases inspect the private ``_headers`` member on an
        OpenTelemetry exporter. That member was removed in newer OpenTelemetry
        releases. When this specific incompatibility is encountered, create an
        equivalent OTLP HTTP provider directly instead.
        """
        try:
            return potel.register(
                endpoint=endpoint,
                project_name=project_name,
                set_global_tracer_provider=False,
                auto_instrument=False,
                verbose=False,
                **register_kwargs,
            )
        except AttributeError as exc:
            if "_headers" not in str(exc):
                raise

        warnings.warn(
            "Phoenix's registration helper is incompatible with the installed "
            "OpenTelemetry exporter; using direct OTLP HTTP export instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        headers = register_kwargs.get("headers")
        provider = TracerProvider(
            resource=Resource.create({"openinference.project.name": project_name})
        )
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
        return provider

    def _start_span(self, event: TelemetryEvent) -> None:
        super()._start_span(event)
        with self._lock:
            span = self._active_spans.get(event.trace_context.span_id)

        if span is None:
            return

        # OpenInference specific enrichments
        if isinstance(event, AgentRunStartEvent):
            if self._capture_content and event.user_prompt:
                span.set_attribute("input.value", event.user_prompt)

        elif isinstance(event, LLMCallStartEvent):
            span.set_attribute("llm.model_name", event.model)
            if self._capture_content:
                if event.input_prompt:
                    span.set_attribute("input.value", event.input_prompt)
                elif event.input_messages:
                    span.set_attribute("input.value", json.dumps(list(event.input_messages), ensure_ascii=False))

                if event.input_messages:
                    for idx, msg in enumerate(event.input_messages):
                        if isinstance(msg, dict):
                            prefix = f"llm.input_messages.{idx}.message"
                            if "role" in msg and msg["role"]:
                                span.set_attribute(f"{prefix}.role", str(msg["role"]))
                            if "content" in msg and msg["content"]:
                                span.set_attribute(f"{prefix}.content", str(msg["content"]))

        elif isinstance(event, ToolCallStartEvent):
            span.set_attribute("tool.name", event.tool_name)
            if self._capture_content:
                args_str = json.dumps(event.arguments) if event.arguments else "{}"
                span.set_attribute("tool.parameters", args_str)
                span.set_attribute("input.value", args_str)

        elif isinstance(event, MCPCallStartEvent):
            span.set_attribute("tool.name", f"{event.server_name}.{event.tool_name}")
            if self._capture_content:
                args_str = json.dumps(event.arguments) if event.arguments else "{}"
                span.set_attribute("tool.parameters", args_str)
                span.set_attribute("input.value", args_str)

    def _end_span(self, event: TelemetryEvent) -> None:
        with self._lock:
            span = self._active_spans.get(event.trace_context.span_id)

        if span is not None:
            # Set OpenInference output and token attributes before span closes
            if isinstance(event, AgentRunEndEvent):
                if self._capture_content and event.output_text:
                    span.set_attribute("output.value", event.output_text)

            elif isinstance(event, LLMCallEndEvent):
                if self._capture_content:
                    if event.output_text:
                        span.set_attribute("output.value", event.output_text)
                    elif event.output_tool_calls:
                        span.set_attribute(
                            "output.value",
                            json.dumps(list(event.output_tool_calls), ensure_ascii=False),
                        )

                    # OpenInference structured output message attributes
                    if event.output_text or event.output_tool_calls:
                        span.set_attribute("llm.output_messages.0.message.role", "assistant")
                        if event.output_text:
                            span.set_attribute("llm.output_messages.0.message.content", event.output_text)
                        for idx, tc in enumerate(event.output_tool_calls):
                            prefix = f"llm.output_messages.0.message.tool_calls.{idx}.tool_call"
                            if isinstance(tc, dict):
                                if tc.get("id"):
                                    span.set_attribute(f"{prefix}.id", str(tc["id"]))
                                if tc.get("name"):
                                    span.set_attribute(f"{prefix}.function.name", str(tc["name"]))
                                if "arguments" in tc:
                                    args = tc["arguments"]
                                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                                    span.set_attribute(f"{prefix}.function.arguments", args_str)

                if event.token_usage:
                    if event.token_usage.prompt_tokens > 0:
                        span.set_attribute("llm.token_count.prompt", event.token_usage.prompt_tokens)
                    if event.token_usage.completion_tokens > 0:
                        span.set_attribute("llm.token_count.completion", event.token_usage.completion_tokens)
                    if event.token_usage.total_tokens > 0:
                        span.set_attribute("llm.token_count.total", event.token_usage.total_tokens)

            elif isinstance(event, ToolCallEndEvent):
                if self._capture_content and event.result is not None:
                    span.set_attribute("output.value", str(event.result))

            elif isinstance(event, MCPCallEndEvent):
                if self._capture_content:
                    span.set_attribute("output.value", str(event.status))

        super()._end_span(event)

