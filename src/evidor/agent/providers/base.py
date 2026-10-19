import asyncio
import json
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from evidor.models import GenerationRequest, GenerationResponse
from evidor.utils import run_async_in_sync

if TYPE_CHECKING:
    from evidor.tools.tool import Tool


def is_client_closed(client: Any) -> bool:
    """Safely check if an SDK or HTTP client has been closed."""
    if client is None:
        return True
    is_closed_attr = getattr(client, "is_closed", None)
    if is_closed_attr is not None:
        if callable(is_closed_attr):
            try:
                res = is_closed_attr()
                if isinstance(res, bool):
                    return res
            except Exception:
                pass
        elif isinstance(is_closed_attr, bool):
            return is_closed_attr

    sub_client = getattr(client, "_client", None)
    if sub_client is not None:
        sub_is_closed = getattr(sub_client, "is_closed", None)
        if sub_is_closed is not None:
            if callable(sub_is_closed):
                try:
                    res = sub_is_closed()
                    if isinstance(res, bool):
                        return res
                except Exception:
                    pass
            elif isinstance(sub_is_closed, bool):
                return sub_is_closed
    return False


@runtime_checkable
class ModelProvider(Protocol):
    """Produces one normalized response from one normalized request."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the request synchronously."""
        ...

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the request asynchronously."""
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
        self._lock = threading.Lock()

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

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response asynchronously. Subclasses should override this natively."""
        if type(self).generate != BaseProvider.generate:
            return await asyncio.to_thread(self.generate, request)
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_async")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response synchronously. Dispatches to generate_async via run_async_in_sync."""
        return run_async_in_sync(
            lambda: self.generate_async(request),
            error_message=(
                f"{self.__class__.__name__}.generate() cannot be called from a running event loop. "
                "Use 'await provider.generate_async()' instead."
            ),
        )

    def close(self) -> None:
        """Close any cached synchronous client sessions."""
        pass

    async def close_async(self) -> None:
        """Close any cached asynchronous or synchronous client sessions."""
        self.close()


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
