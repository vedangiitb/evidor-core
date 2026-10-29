import asyncio
import json
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

from evidor.constants import format_provider_generate_error
from evidor.models import (
    AsyncStreamResponse,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    StreamResponse,
    ToolCall,
)
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

    @overload
    def generate(self, request: GenerationRequest, *, stream: Literal[False] = False) -> GenerationResponse: ...

    @overload
    def generate(self, request: GenerationRequest, *, stream: Literal[True]) -> StreamResponse: ...

    @overload
    def generate(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse: ...

    def generate(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse:
        """Generate a response for the request synchronously."""
        ...

    @overload
    async def generate_async(
        self, request: GenerationRequest, *, stream: Literal[False] = False
    ) -> GenerationResponse: ...

    @overload
    async def generate_async(
        self, request: GenerationRequest, *, stream: Literal[True]
    ) -> AsyncStreamResponse: ...

    @overload
    async def generate_async(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse: ...

    async def generate_async(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse:
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

    @property
    def base_url(self) -> str | None:
        """The custom base URL for the model provider endpoint, if configured."""
        return getattr(self, "_base_url", None)

    async def generate_async(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse:
        """Generate a response asynchronously. Subclasses should override this natively."""
        if type(self).generate != BaseProvider.generate:
            return await asyncio.to_thread(self.generate, request, stream=stream)
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_async")

    def generate(
        self, request: GenerationRequest, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse:
        """Generate a response synchronously. Dispatches to generate_async via run_async_in_sync."""
        return run_async_in_sync(
            lambda: self.generate_async(request, stream=stream),
            error_message=format_provider_generate_error(self.__class__.__name__),
        )

    def close(self) -> None:
        """Close any cached synchronous client sessions."""
        pass

    async def close_async(self) -> None:
        """Close any cached asynchronous or synchronous client sessions."""
        self.close()


class ToolCallAccumulator:
    """Accumulates in-flight streaming tool calls across partial chunks."""

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, Any]] = {}
        self._completed: list[ToolCall] = []

    def start_call(self, call_id: str, name: str) -> None:
        if call_id not in self._calls:
            self._calls[call_id] = {"id": call_id, "name": name, "args_parts": []}
        elif name and not self._calls[call_id]["name"]:
            self._calls[call_id]["name"] = name

    def add_arguments_delta(self, call_id: str, delta: str) -> None:
        if call_id in self._calls:
            self._calls[call_id]["args_parts"].append(delta)
        else:
            self._calls[call_id] = {"id": call_id, "name": "", "args_parts": [delta]}

    def complete_call(self, call_id: str) -> ToolCall | None:
        data = self._calls.pop(call_id, None)
        if not data:
            return None
        raw_args = "".join(data["args_parts"])
        parsed = safe_parse_tool_arguments(raw_args)
        tc = ToolCall(id=data["id"], name=data["name"] or "tool", arguments=parsed)
        self._completed.append(tc)
        return tc

    def finalize_all(self) -> list[ToolCall]:
        for call_id in list(self._calls.keys()):
            self.complete_call(call_id)
        return list(self._completed)


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
    return sorted(
        tools,
        key=lambda t: (t.get("name") if isinstance(t, dict) else getattr(t, "name", "")) or "",
    )


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
