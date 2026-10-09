"""Prometheus metrics adapter exporting Evidor operational metrics.

Exports standard agent metrics (runs, latency, token counts, tool invocations, errors)
to Prometheus collectors and registries.
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
    MCPCallEndEvent,
    TelemetryEvent,
    ToolCallEndEvent,
)
from ..sink import TelemetrySink

try:
    import prometheus_client
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, REGISTRY

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    CollectorRegistry = Any  # type: ignore[assignment, misc]
    Counter = Any  # type: ignore[assignment, misc]
    Gauge = Any  # type: ignore[assignment, misc]
    Histogram = Any  # type: ignore[assignment, misc]
    REGISTRY = None  # type: ignore[assignment]


def _get_or_create_collector(
    metric_cls: type,
    name: str,
    documentation: str,
    labelnames: Sequence[str],
    registry: Any,
) -> Any:
    """Retrieve an existing metric collector from the registry if present, or create a new one."""
    if registry is not None and hasattr(registry, "_names_to_collectors"):
        # prometheus_client registers both 'name' and sometimes 'name_total'
        collector = registry._names_to_collectors.get(name) or registry._names_to_collectors.get(f"{name}_total")
        if collector is not None:
            return collector
    return metric_cls(name, documentation, labelnames=labelnames, registry=registry)


class PrometheusSink(TelemetrySink):
    """Telemetry sink exporting Evidor metrics to Prometheus."""

    def __init__(
        self,
        registry: Any = None,
        metric_prefix: str = "evidor",
    ) -> None:
        if not HAS_PROMETHEUS:
            raise ImportError(
                "The 'prometheus-client' package is required for the Prometheus adapter. "
                "Install it with: pip install 'evidor[prometheus]'"
            )

        self._registry = registry or REGISTRY
        self._prefix = metric_prefix.rstrip("_")
        self._lock = threading.Lock()

        # 1. Agent runs
        self.agent_runs_total = _get_or_create_collector(
            Counter,
            f"{self._prefix}_agent_runs_total",
            "Total number of agent runs completed",
            ["agent_name", "status"],
            self._registry,
        )
        self.agent_run_duration_seconds = _get_or_create_collector(
            Histogram,
            f"{self._prefix}_agent_run_duration_seconds",
            "Duration of agent execution turns in seconds",
            ["agent_name"],
            self._registry,
        )
        self.active_agent_runs = _get_or_create_collector(
            Gauge,
            f"{self._prefix}_active_agent_runs",
            "Number of currently active agent runs",
            ["agent_name"],
            self._registry,
        )

        # 2. LLM calls
        self.llm_calls_total = _get_or_create_collector(
            Counter,
            f"{self._prefix}_llm_calls_total",
            "Total number of LLM provider calls",
            ["provider", "model", "status"],
            self._registry,
        )
        self.llm_call_duration_seconds = _get_or_create_collector(
            Histogram,
            f"{self._prefix}_llm_call_duration_seconds",
            "Duration of LLM calls in seconds",
            ["provider", "model"],
            self._registry,
        )
        self.tokens_total = _get_or_create_collector(
            Counter,
            f"{self._prefix}_tokens_total",
            "Total LLM tokens consumed",
            ["provider", "model", "token_type"],
            self._registry,
        )

        # 3. Tool calls
        self.tool_calls_total = _get_or_create_collector(
            Counter,
            f"{self._prefix}_tool_calls_total",
            "Total number of local tool executions",
            ["tool_name", "status"],
            self._registry,
        )
        self.tool_call_duration_seconds = _get_or_create_collector(
            Histogram,
            f"{self._prefix}_tool_call_duration_seconds",
            "Duration of tool executions in seconds",
            ["tool_name"],
            self._registry,
        )

        # 4. MCP calls
        self.mcp_calls_total = _get_or_create_collector(
            Counter,
            f"{self._prefix}_mcp_calls_total",
            "Total number of MCP tool calls",
            ["server_name", "tool_name", "status"],
            self._registry,
        )
        self.mcp_call_duration_seconds = _get_or_create_collector(
            Histogram,
            f"{self._prefix}_mcp_call_duration_seconds",
            "Duration of MCP tool calls in seconds",
            ["server_name", "tool_name"],
            self._registry,
        )

        # 5. Errors
        self.errors_total = _get_or_create_collector(
            Counter,
            f"{self._prefix}_errors_total",
            "Total number of errors encountered",
            ["error_type"],
            self._registry,
        )

    @property
    def registry(self) -> Any:
        """Return the Prometheus collector registry."""
        return self._registry

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        """Process incoming events and update corresponding Prometheus metrics."""
        for event in events:
            self._record_event(event)

    def _record_event(self, event: TelemetryEvent) -> None:
        if isinstance(event, AgentRunStartEvent):
            agent_name = event.attributes.get("agent_name", "default")
            self.active_agent_runs.labels(agent_name=agent_name).inc()

        elif isinstance(event, AgentRunEndEvent):
            agent_name = event.attributes.get("agent_name", "default")
            self.active_agent_runs.labels(agent_name=agent_name).dec()
            self.agent_runs_total.labels(agent_name=agent_name, status=event.status).inc()
            if event.duration_ms > 0:
                self.agent_run_duration_seconds.labels(agent_name=agent_name).observe(event.duration_ms / 1000.0)

        elif isinstance(event, LLMCallEndEvent):
            provider = event.provider or "unknown"
            model = event.model or "unknown"
            self.llm_calls_total.labels(provider=provider, model=model, status=event.status).inc()
            if event.duration_ms > 0:
                self.llm_call_duration_seconds.labels(provider=provider, model=model).observe(event.duration_ms / 1000.0)

            if event.token_usage:
                if event.token_usage.prompt_tokens > 0:
                    self.tokens_total.labels(provider=provider, model=model, token_type="prompt").inc(
                        event.token_usage.prompt_tokens
                    )
                if event.token_usage.completion_tokens > 0:
                    self.tokens_total.labels(provider=provider, model=model, token_type="completion").inc(
                        event.token_usage.completion_tokens
                    )
                if event.token_usage.total_tokens > 0:
                    self.tokens_total.labels(provider=provider, model=model, token_type="total").inc(
                        event.token_usage.total_tokens
                    )

        elif isinstance(event, ToolCallEndEvent):
            tool_name = event.tool_name or "unknown"
            self.tool_calls_total.labels(tool_name=tool_name, status=event.status).inc()
            if event.duration_ms > 0:
                self.tool_call_duration_seconds.labels(tool_name=tool_name).observe(event.duration_ms / 1000.0)

        elif isinstance(event, MCPCallEndEvent):
            server_name = event.server_name or "unknown"
            tool_name = event.tool_name or "unknown"
            self.mcp_calls_total.labels(server_name=server_name, tool_name=tool_name, status=event.status).inc()
            if event.duration_ms > 0:
                self.mcp_call_duration_seconds.labels(server_name=server_name, tool_name=tool_name).observe(
                    event.duration_ms / 1000.0
                )

        elif isinstance(event, ErrorEvent):
            error_type = event.error_type or "unknown"
            self.errors_total.labels(error_type=error_type).inc()

    def export_text(self) -> str:
        """Export current Prometheus metrics as Prometheus exposition text format."""
        if not HAS_PROMETHEUS:
            return ""
        return prometheus_client.generate_latest(self._registry).decode("utf-8")

    def flush(self) -> None:
        """No-op for in-process Prometheus collectors."""
        pass

    def close(self) -> None:
        """No-op for in-process Prometheus collectors."""
        pass
