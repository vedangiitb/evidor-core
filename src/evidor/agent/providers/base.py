"""The port and base class implemented by model-provider adapters."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from evidor.models import GenerationRequest, GenerationResponse

if TYPE_CHECKING:
    from evidor.tools.tool import Tool


@runtime_checkable
class ModelProvider(Protocol):
    """Produces one normalized response from one normalized request."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the request."""
        ...

    def with_model(self, model: str) -> "ModelProvider":
        """Return a copy of this provider configured with a different model."""
        ...


class BaseProvider:
    """Common base class for concrete model providers."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking: Any = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._thinking = thinking

    @property
    def model(self) -> str:
        """The identifier of the active model."""
        return self._model

    @property
    def thinking(self) -> Any:
        """The active thinking configuration, if any."""
        return self._thinking

    @property
    def context_strategy(self) -> str:
        """The native context management strategy implemented by this provider."""
        return "stateless_replay"

    @property
    def compact_threshold(self) -> int | None:
        """The token threshold for provider-native context compaction, if configured."""
        return getattr(self, "_compact_threshold", None)


def extract_system_prompt(request: GenerationRequest) -> str:
    """Extract and combine system instructions from request.system and system messages."""
    prompts: list[str] = []
    if request.system:
        prompts.append(request.system)
    for message in request.messages:
        if message.role == "system" and message.content:
            prompts.append(message.content)
    return "\n\n".join(prompts)


def sort_tools(tools: Sequence[Any] | None) -> list[Any]:
    """Sort tools alphabetically by name for deterministic ordering and prompt caching."""
    if not tools:
        return []
    return sorted(tools, key=lambda t: getattr(t, "name", ""))


def safe_parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    """Safely decode tool call arguments from a dictionary, JSON string, or object."""
    if isinstance(raw_args, dict):
        return raw_args
    if hasattr(raw_args, "items"):
        try:
            return dict(raw_args.items())
        except Exception:
            pass
    if not isinstance(raw_args, str):
        return {}
    try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict):
            return parsed
        return {"__raw_args__": parsed}
    except Exception as err:
        return {"__raw_args__": raw_args, "__decode_error__": str(err)}


def build_token_usage(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> Any:
    """Safely construct a normalized TokenUsage instance."""
    try:
        from evidor.telemetry.events import TokenUsage

        tt = total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=tt,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
    except Exception:
        return None
