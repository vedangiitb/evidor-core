"""Anthropic Claude provider adapter."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, overload

from evidor.constants import ANTHROPIC_GENERATE_SYNC_ERROR
from evidor.models import (
    AsyncStreamResponse,
    GenerationRequest,
    GenerationResponse,
    Message,
    StreamChunk,
    StreamResponse,
    ToolCall,
)
from evidor.utils import check_sync_allowed
from ..base import BaseProvider, is_client_closed
from ..client_pool import close_async_clients, close_sync_client, get_or_create_async_client
from .builder import AnthropicPayloadBuilder
from .parser import AnthropicResponseParser
from .stream import AnthropicStreamHandler


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic Claude using the Messages API.

    Context Management:
        Anthropic's Messages API is fundamentally stateless: it does not store conversation
        threads or chain from a conversation ID on the server. Evidor's Agent manages
        multi-turn conversations by replaying the conversational history (`stateless_replay`).

        When `enable_prompt_caching=True` (default), Evidor includes ephemeral prompt caching
        headers (`cache_control={"type": "ephemeral"}`), allowing Anthropic to cache the
        prompt prefix (system prompt, tools, and prior turns) in ephemeral KV-cache, reducing
        input latency and cost by up to 90% without requiring server-side conversation state.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
        enable_prompt_caching: bool = True,
        thinking: Any = None,
        compact_threshold: int | None = None,
        compact_instructions: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if thinking is not None:
            from evidor.models import AnthropicThinkingConfig

            thinking_cfg: AnthropicThinkingConfig | None = (
                thinking
                if isinstance(thinking, AnthropicThinkingConfig)
                else AnthropicThinkingConfig.from_value(thinking)
            )
            thinking_cfg.validate()
        else:
            thinking_cfg = None

        super().__init__(model=model, api_key=api_key, thinking=thinking_cfg)
        self._enable_prompt_caching = enable_prompt_caching
        self._compact_threshold = compact_threshold
        self._compact_instructions = compact_instructions
        self._base_url = base_url
        self._sync_client: Any = None
        self._default_async_client: Any = None
        self._async_clients: dict[asyncio.AbstractEventLoop, Any] = {}

        if max_tokens is None:
            if self._thinking and self._thinking.enabled and self._thinking.budget_tokens:
                max_tokens = self._thinking.budget_tokens + 2048
            else:
                max_tokens = 1024
        self._max_tokens = max_tokens

        self._builder = AnthropicPayloadBuilder(
            model=self._model,
            max_tokens=self._max_tokens,
            enable_prompt_caching=self._enable_prompt_caching,
            thinking=self._thinking,
            compact_threshold=self._compact_threshold,
            compact_instructions=self._compact_instructions,
        )
        self._parser = AnthropicResponseParser(model=self._model)

    @property
    def context_strategy(self) -> str:
        return "stateless_replay"

    @property
    def enable_prompt_caching(self) -> bool:
        return self._enable_prompt_caching

    @property
    def compact_threshold(self) -> int | None:
        return self._compact_threshold

    @property
    def compact_instructions(self) -> str | None:
        return self._compact_instructions

    @property
    def base_url(self) -> str | None:
        """The custom base URL for Anthropic-compatible endpoints, if configured."""
        return self._base_url

    def with_model(self, model: str) -> "AnthropicProvider":
        return AnthropicProvider(
            model=model,
            api_key=self._api_key,
            max_tokens=self._max_tokens,
            enable_prompt_caching=self._enable_prompt_caching,
            thinking=self._thinking,
            compact_threshold=self._compact_threshold,
            compact_instructions=self._compact_instructions,
            base_url=self._base_url,
        )

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
        check_sync_allowed(ANTHROPIC_GENERATE_SYNC_ERROR)

        client = self._get_client()
        should_stream = stream or request.stream
        payload = self._builder.build(request)
        if should_stream:
            payload["stream"] = True
            if self._compact_threshold is not None or self._compact_instructions is not None:
                raw_stream = client.beta.messages.create(**payload)
            else:
                raw_stream = client.messages.create(**payload)
            return StreamResponse(self._create_stream(raw_stream))

        if self._compact_threshold is not None or self._compact_instructions is not None:
            response_msg = client.beta.messages.create(**payload)
        else:
            response_msg = client.messages.create(**payload)
        return self._parser.parse(response_msg)

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
        client = self._get_async_client()
        should_stream = stream or request.stream
        payload = self._builder.build(request)
        if should_stream:
            payload["stream"] = True
            if self._compact_threshold is not None or self._compact_instructions is not None:
                raw_stream = await client.beta.messages.create(**payload)
            else:
                raw_stream = await client.messages.create(**payload)
            return AsyncStreamResponse(self._create_async_stream(raw_stream))

        if self._compact_threshold is not None or self._compact_instructions is not None:
            response_msg = await client.beta.messages.create(**payload)
        else:
            response_msg = await client.messages.create(**payload)
        return self._parser.parse(response_msg)

    def _create_stream(self, raw_stream: Any) -> Iterator[StreamChunk]:
        handler = AnthropicStreamHandler(model=self._model, parser=self._parser)
        for event in raw_stream:
            yield from handler.process_event(event)
        yield from handler.finalize()

    async def _create_async_stream(self, raw_stream: Any) -> AsyncIterator[StreamChunk]:
        handler = AnthropicStreamHandler(model=self._model, parser=self._parser)
        async for event in raw_stream:
            for chunk in handler.process_event(event):
                yield chunk
        for chunk in handler.finalize():
            yield chunk

    def close(self) -> None:
        with self._lock:
            client = self._sync_client
            self._sync_client = None
        close_sync_client(client)

    async def close_async(self) -> None:
        self.close()
        with self._lock:
            clients_map = getattr(self, "_async_clients", {})
            default_client = self._default_async_client
            self._default_async_client = None
        await close_async_clients(clients_map, default_client, self._lock)

    def _get_client(self) -> Any:
        with self._lock:
            if not is_client_closed(self._sync_client):
                return self._sync_client

            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise ImportError("Install Anthropic support with: pip install 'evidor[anthropic]'") from error

            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url is not None:
                client_kwargs["base_url"] = self._base_url
            client = Anthropic(**client_kwargs)
            self._sync_client = client
            return client

    def _get_async_client(self) -> Any:
        try:
            from anthropic import AsyncAnthropic
        except (ImportError, AttributeError) as error:
            raise ImportError("Install Anthropic support with: pip install 'evidor[anthropic]'") from error

        if not hasattr(self, "_async_clients"):
            self._async_clients = {}

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url

        client, self._default_async_client = get_or_create_async_client(
            self._async_clients,
            self._default_async_client,
            lambda: AsyncAnthropic(**client_kwargs),
            self._lock,
        )
        return client

    # Backward-compatible delegates
    def _format_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return self._builder.format_messages(messages)
