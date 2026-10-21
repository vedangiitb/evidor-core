"""Google Gemini provider adapter."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, overload

from evidor.constants import GEMINI_GENERATE_SYNC_ERROR
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
from ..base import BaseProvider
from ..client_pool import close_sync_client
from .builder import GeminiPayloadBuilder
from .parser import GeminiResponseParser
from .stream import GeminiStreamHandler


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini using the Interactions API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cached_content: str | None = None,
        thinking: Any = None,
    ) -> None:
        if thinking is not None:
            from evidor.models import GeminiThinkingConfig

            thinking_cfg: GeminiThinkingConfig | None = (
                thinking
                if isinstance(thinking, GeminiThinkingConfig)
                else GeminiThinkingConfig.from_value(thinking)
            )
            thinking_cfg.validate()
        else:
            thinking_cfg = None

        super().__init__(model=model, api_key=api_key, thinking=thinking_cfg)
        self._cached_content = cached_content
        self._client: Any = None
        self._types: Any = None

        self._builder = GeminiPayloadBuilder(
            model=self._model,
            cached_content=self._cached_content,
            thinking=self._thinking,
        )
        self._parser = GeminiResponseParser(model=self._model)

    @property
    def context_strategy(self) -> str:
        return "interaction_chain"

    @property
    def cached_content(self) -> str | None:
        return self._cached_content

    def with_model(self, model: str) -> "GeminiProvider":
        return GeminiProvider(
            model=model,
            api_key=self._api_key,
            cached_content=self._cached_content,
            thinking=self._thinking,
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
        check_sync_allowed(GEMINI_GENERATE_SYNC_ERROR)

        client, types = self._get_client_and_types()
        should_stream = stream or request.stream
        payload = self._builder.build(request, types)
        if should_stream:
            payload["stream"] = True
            raw_stream = client.interactions.create(**payload)
            return StreamResponse(self._create_stream(raw_stream))

        response = client.interactions.create(**payload)
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
        client, types = self._get_client_and_types()
        should_stream = stream or request.stream
        payload = self._builder.build(request, types)
        if hasattr(client, "aio") and hasattr(client.aio, "interactions") and hasattr(client.aio.interactions, "create"):
            if should_stream:
                payload["stream"] = True
                raw_stream = await client.aio.interactions.create(**payload)
                return AsyncStreamResponse(self._create_async_stream(raw_stream))

            response = await client.aio.interactions.create(**payload)
            return self._parser.parse(response)
        return await asyncio.to_thread(self.generate, request, stream=stream)

    def _create_stream(self, raw_stream: Any) -> Iterator[StreamChunk]:
        handler = GeminiStreamHandler(self._parser)
        for event in raw_stream:
            yield from handler.process_event(event)
        yield from handler.finalize()

    async def _create_async_stream(self, raw_stream: Any) -> AsyncIterator[StreamChunk]:
        handler = GeminiStreamHandler(self._parser)
        async for event in raw_stream:
            for chunk in handler.process_event(event):
                yield chunk
        for chunk in handler.finalize():
            yield chunk

    def close(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._types = None
        close_sync_client(client)

    async def close_async(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._types = None

        if client is not None:
            aio_client = getattr(client, "aio", None)
            close_async_fn = getattr(aio_client, "close", None)
            if callable(close_async_fn):
                try:
                    await close_async_fn()
                except Exception:
                    pass
            close_sync_client(client)

    def _get_client_and_types(self) -> tuple[Any, Any]:
        with self._lock:
            if getattr(self, "_client", None) is not None and getattr(self, "_types", None) is not None:
                return self._client, self._types

            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise ImportError("Install Gemini support with: pip install 'evidor[gemini]'") from error

            client = genai.Client(api_key=self._api_key)
            if not hasattr(client, "interactions") or not hasattr(client.interactions, "create"):
                raise RuntimeError(
                    "Modern Google GenAI Interactions API (client.interactions.create) is required. "
                    "Please update google-genai to the latest version."
                )
            self._client = client
            self._types = types
            return self._client, self._types

