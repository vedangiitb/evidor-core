"""OpenAI provider adapter."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, overload

from evidor.constants import OPENAI_GENERATE_SYNC_ERROR
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
from .builder import OpenAIPayloadBuilder
from .parser import OpenAIResponseParser
from .stream import OpenAIStreamHandler


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI using the modern Responses API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking: Any = None,
        compact_threshold: int | None = None,
        base_url: str | None = None,
    ) -> None:
        if thinking is not None:
            from evidor.models import OpenAIThinkingConfig

            thinking_cfg: OpenAIThinkingConfig | None = (
                thinking
                if isinstance(thinking, OpenAIThinkingConfig)
                else OpenAIThinkingConfig.from_value(thinking)
            )
            thinking_cfg.validate()
        else:
            thinking_cfg = None

        super().__init__(model=model, api_key=api_key, thinking=thinking_cfg)
        self._compact_threshold = compact_threshold
        self._base_url = base_url
        self._sync_client: Any = None
        self._default_async_client: Any = None
        self._async_clients: dict[asyncio.AbstractEventLoop, Any] = {}

        self._builder = OpenAIPayloadBuilder(
            model=self._model,
            thinking=self._thinking,
            compact_threshold=self._compact_threshold,
        )
        self._parser = OpenAIResponseParser(model=self._model)

    @property
    def context_strategy(self) -> str:
        return "response_chain"

    @property
    def compact_threshold(self) -> int | None:
        return self._compact_threshold

    @property
    def base_url(self) -> str | None:
        """The custom base URL for OpenAI-compatible endpoints, if configured."""
        return self._base_url

    def with_model(self, model: str) -> "OpenAIProvider":
        return OpenAIProvider(
            model=model,
            api_key=self._api_key,
            thinking=self._thinking,
            compact_threshold=self._compact_threshold,
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
        check_sync_allowed(OPENAI_GENERATE_SYNC_ERROR)

        client = self._get_client()
        should_stream = stream or request.stream
        payload = self._builder.build(request)
        if should_stream:
            payload["stream"] = True
            raw_stream = client.responses.create(**payload)
            return StreamResponse(self._create_stream(raw_stream))

        response = client.responses.create(**payload)
        return self._parser.parse(response)

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
        if client is not None:
            payload = self._builder.build(request)
            if should_stream:
                payload["stream"] = True
                raw_stream = await client.responses.create(**payload)
                return AsyncStreamResponse(self._create_async_stream(raw_stream))

            response = await client.responses.create(**payload)
            return self._parser.parse(response)
        return await asyncio.to_thread(self.generate, request, stream=stream)

    def _create_stream(self, raw_stream: Any) -> Iterator[StreamChunk]:
        handler = OpenAIStreamHandler(self._parser)
        for event in raw_stream:
            yield from handler.process_event(event)
        yield from handler.finalize()

    async def _create_async_stream(self, raw_stream: Any) -> AsyncIterator[StreamChunk]:
        handler = OpenAIStreamHandler(self._parser)
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
                from openai import OpenAI
            except ImportError as error:
                raise ImportError("Install OpenAI support with: pip install 'evidor[openai]'") from error

            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url is not None:
                client_kwargs["base_url"] = self._base_url
            client = OpenAI(**client_kwargs)
            if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
                raise RuntimeError(
                    "Modern OpenAI Responses API (client.responses.create) is required. "
                    "Please update openai to the latest version."
                )
            self._sync_client = client
            return client

    def _get_async_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except (ImportError, AttributeError):
            return None

        if not hasattr(self, "_async_clients"):
            self._async_clients = {}

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url

        client, self._default_async_client = get_or_create_async_client(
            self._async_clients,
            self._default_async_client,
            lambda: AsyncOpenAI(**client_kwargs),
            self._lock,
        )

        if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
            return None
        return client



