"""Thinking and reasoning configurations for models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ThinkingEffort = str


@dataclass(frozen=True, slots=True)
class ThinkingConfig:
    """Base class for model thinking / reasoning configuration."""

    enabled: bool = True

    def validate(self) -> None:
        """Hook for provider-neutral constraint checks."""
        pass


@dataclass(frozen=True, slots=True)
class OpenAIThinkingConfig(ThinkingConfig):
    """Configuration for OpenAI reasoning models (GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, etc.)."""

    effort: ThinkingEffort = "medium"
    summary: str | None = None

    def validate(self) -> None:
        pass

    @classmethod
    def from_value(cls, value: Any) -> OpenAIThinkingConfig:
        if isinstance(value, OpenAIThinkingConfig):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, str):
            return cls(enabled=True, effort=value)
        raise TypeError(f"Cannot coerce {type(value).__name__} to OpenAIThinkingConfig")


@dataclass(frozen=True, slots=True)
class AnthropicThinkingConfig(ThinkingConfig):
    """Configuration for Anthropic extended thinking (Claude 3.7 Sonnet, Claude 4+)."""

    budget_tokens: int | None = 2048
    type: str = "enabled"
    effort: str | None = None

    def validate(self) -> None:
        pass

    @classmethod
    def from_value(cls, value: Any) -> AnthropicThinkingConfig:
        if isinstance(value, AnthropicThinkingConfig):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, int):
            return cls(enabled=True, type="enabled", budget_tokens=value)
        if isinstance(value, str):
            if value == "adaptive":
                return cls(enabled=True, type="adaptive", budget_tokens=None)
            return cls(enabled=True, type="adaptive", effort=value, budget_tokens=None)
        raise TypeError(f"Cannot coerce {type(value).__name__} to AnthropicThinkingConfig")


@dataclass(frozen=True, slots=True)
class GeminiThinkingConfig:
    """Configuration for Google Gemini thinking models in the Interactions API."""

    thinking_level: str | None = None  # Supported in Gemini 3.x Interactions API: 'minimal', 'low', 'medium', 'high'
    thinking_summaries: str | None = None  # Supported in Gemini Interactions API: 'auto', 'none'
    thinking_budget: int | None = None  # Supported in earlier Gemini models (e.g. Gemini 2.5)

    def validate(self) -> None:
        pass

    @classmethod
    def from_value(cls, value: Any) -> GeminiThinkingConfig:
        if isinstance(value, GeminiThinkingConfig):
            return value
        if isinstance(value, bool):
            raise TypeError(
                "Cannot coerce bool to GeminiThinkingConfig. "
                "Gemini Interactions API supports thinking_level (e.g. 'minimal', 'low', 'medium', 'high'), "
                "thinking_summaries ('auto', 'none'), or thinking_budget (int)."
            )
        if isinstance(value, int):
            return cls(thinking_budget=value)
        if isinstance(value, str):
            return cls(thinking_level=value)
        raise TypeError(
            f"Cannot coerce {type(value).__name__} to GeminiThinkingConfig. "
            "Gemini Interactions API supports thinking_level (e.g. 'minimal', 'low', 'medium', 'high'), "
            "thinking_summaries ('auto', 'none'), or thinking_budget (int)."
        )


AnyThinkingConfig = ThinkingConfig | OpenAIThinkingConfig | AnthropicThinkingConfig | GeminiThinkingConfig

