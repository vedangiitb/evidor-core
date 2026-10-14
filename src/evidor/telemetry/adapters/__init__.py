"""Telemetry provider adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .langfuse import LangfuseSink
    from .opentelemetry import OpenTelemetrySink
    from .phoenix import PhoenixSink
    from .prometheus import PrometheusSink

_ADAPTER_MAP = {
    "LangfuseSink": ".langfuse",
    "OpenTelemetrySink": ".opentelemetry",
    "PhoenixSink": ".phoenix",
    "PrometheusSink": ".prometheus",
}

__all__ = [
    "LangfuseSink",
    "OpenTelemetrySink",
    "PhoenixSink",
    "PrometheusSink",
]


def __getattr__(name: str) -> Any:
    if name in _ADAPTER_MAP:
        import importlib

        module = importlib.import_module(_ADAPTER_MAP[name], __package__)
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
