"""Prometheus metrics adapter exporting Evidor operational metrics."""

from __future__ import annotations

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
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    REGISTRY = None  # type: ignore[assignment]


class PrometheusSink(TelemetrySink):
    """Telemetry sink exporting Evidor metrics to Prometheus."""

    def __init__(
        self,
        registry: Any = None,
        metric_prefix: str = "evidor",
        capture_content: bool = True,
    ) -> None:
        if not HAS_PROMETHEUS:
            raise ImportError(
                "The 'prometheus-client' package is required for the Prometheus adapter. "
                "Install it with: pip install 'evidor[prometheus]'"
            )

        self._registry = registry or REGISTRY
        pfx = metric_prefix.rstrip("_")

        def _metric(cls: type, name: str, doc: str, labels: list[str]) -> Any:
            full_name = f"{pfx}_{name}"
            reg = self._registry
            if reg is not None and hasattr(reg, "_names_to_collectors"):
                collector = reg._names_to_collectors.get(full_name) or reg._names_to_collectors.get(f"{full_name}_total")
                if collector is not None:
                    return collector
            return cls(full_name, doc, labelnames=labels, registry=reg)

        self.agent_runs_total = _metric(Counter, "agent_runs_total", "Total agent runs completed", ["agent_name", "status"])
        self.agent_run_duration_seconds = _metric(Histogram, "agent_run_duration_seconds", "Agent run duration (s)", ["agent_name"])
        self.active_agent_runs = _metric(Gauge, "active_agent_runs", "Active agent runs", ["agent_name"])
        self.llm_calls_total = _metric(Counter, "llm_calls_total", "Total LLM calls", ["provider", "model", "status"])
        self.llm_call_duration_seconds = _metric(Histogram, "llm_call_duration_seconds", "LLM call duration (s)", ["provider", "model"])
        self.tokens_total = _metric(Counter, "tokens_total", "Total tokens consumed", ["provider", "model", "token_type"])
        self.tool_calls_total = _metric(Counter, "tool_calls_total", "Total tool calls", ["tool_name", "status"])
        self.tool_call_duration_seconds = _metric(Histogram, "tool_call_duration_seconds", "Tool call duration (s)", ["tool_name"])
        self.mcp_calls_total = _metric(Counter, "mcp_calls_total", "Total MCP calls", ["server_name", "tool_name", "status"])
        self.mcp_call_duration_seconds = _metric(Histogram, "mcp_call_duration_seconds", "MCP call duration (s)", ["server_name", "tool_name"])
        self.errors_total = _metric(Counter, "errors_total", "Total errors", ["error_type"])
        self.llm_retries_total = _metric(Counter, "llm_retries_total", "Total LLM call retries", ["provider", "model"])


    @property
    def registry(self) -> Any:
        return self._registry

    def write(self, events: Sequence[TelemetryEvent]) -> None:
        """Process incoming events and update Prometheus metrics."""
        for event in events:
            self._record(event)

    def _record(self, event: TelemetryEvent) -> None:
        if isinstance(event, AgentRunStartEvent):
            agent = event.attributes.get("agent_name", "default")
            self.active_agent_runs.labels(agent_name=agent).inc()

        elif isinstance(event, AgentRunEndEvent):
            agent = event.attributes.get("agent_name", "default")
            self.active_agent_runs.labels(agent_name=agent).dec()
            self.agent_runs_total.labels(agent_name=agent, status=event.status).inc()
            if event.duration_ms > 0:
                self.agent_run_duration_seconds.labels(agent_name=agent).observe(event.duration_ms / 1000.0)

        elif isinstance(event, LLMCallEndEvent):
            p, m = event.provider or "unknown", event.model or "unknown"
            self.llm_calls_total.labels(provider=p, model=m, status=event.status).inc()
            if event.duration_ms > 0:
                self.llm_call_duration_seconds.labels(provider=p, model=m).observe(event.duration_ms / 1000.0)
            if event.token_usage:
                u = event.token_usage
                for k, count in (("prompt", u.prompt_tokens), ("completion", u.completion_tokens), ("total", u.total_tokens)):
                    if count > 0:
                        self.tokens_total.labels(provider=p, model=m, token_type=k).inc(count)

        elif isinstance(event, ToolCallEndEvent):
            t = event.tool_name or "unknown"
            self.tool_calls_total.labels(tool_name=t, status=event.status).inc()
            if event.duration_ms > 0:
                self.tool_call_duration_seconds.labels(tool_name=t).observe(event.duration_ms / 1000.0)

        elif isinstance(event, MCPCallEndEvent):
            s, t = event.server_name or "unknown", event.tool_name or "unknown"
            self.mcp_calls_total.labels(server_name=s, tool_name=t, status=event.status).inc()
            if event.duration_ms > 0:
                self.mcp_call_duration_seconds.labels(server_name=s, tool_name=t).observe(event.duration_ms / 1000.0)

        elif isinstance(event, ErrorEvent):
            self.errors_total.labels(error_type=event.error_type or "unknown").inc()
            if event.outcome == "retrying":
                p = event.details.get("provider", "unknown")
                m = event.details.get("model", "unknown")
                self.llm_retries_total.labels(provider=p, model=m).inc()


    def export_text(self) -> str:
        """Export current Prometheus metrics as exposition text format."""
        return prometheus_client.generate_latest(self._registry).decode("utf-8") if HAS_PROMETHEUS else ""

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
