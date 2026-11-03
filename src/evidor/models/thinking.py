"""Thinking and reasoning configuration for models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ThinkingEffort = str
ThinkingSummary = str


@dataclass(frozen=True, slots=True)
class ThinkingConfig:
    """Centralized, provider-neutral reasoning and thinking configuration.

    Attributes:
        effort: Desired thinking depth or reasoning level (e.g. 'minimal', 'low', 'medium', 'high', 'xhigh', 'max').
            Passed directly to the provider without client-side mapping or restriction.
        budget_tokens: Token budget for reasoning where supported (e.g. Anthropic extended thinking or earlier Gemini models).
        summary: Reasoning summary exposure (e.g. 'auto', 'concise', 'detailed', 'none').
    """

    effort: ThinkingEffort | None = None
    budget_tokens: int | None = None
    summary: ThinkingSummary | None = None

    def validate(self) -> None:
        """Provider-neutral constraint checks."""
        pass

    @classmethod
    def from_value(cls, value: Any) -> ThinkingConfig:
        """Coerce a value to ThinkingConfig."""
        if isinstance(value, ThinkingConfig):
            return value
        if isinstance(value, str):
            return cls(effort=value)
        if isinstance(value, dict):
            return cls(
                effort=value.get("effort"),
                budget_tokens=value.get("budget_tokens"),
                summary=value.get("summary"),
            )
        raise TypeError(
            f"Cannot coerce {type(value).__name__} to ThinkingConfig. "
            "Thinking must be configured either as an effort string (e.g. thinking='high') "
            "or via ThinkingConfig(effort=..., budget_tokens=..., summary=...)."
        )


AnyThinkingConfig = ThinkingConfig
