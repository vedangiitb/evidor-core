"""Streaming response wrappers and iterators."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
import inspect
import time
from typing import Any

from .generation import GenerationResponse, ToolCall


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A normalized chunk emitted during streaming generation."""

    text: str = ""
    thinking: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    is_finished: bool = False
    response: GenerationResponse | None = None
    raw_event: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.tool_calls, (list, tuple)):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def __str__(self) -> str:
        """Allow print(chunk, end='') directly."""
        return self.text


class StreamResponse(Iterator[StreamChunk]):
    """Synchronous iterator and context manager wrapping a streaming model generation."""

    def __init__(
        self,
        iterator: Iterator[StreamChunk],
        *,
        on_first_token: Callable[[float], None] | None = None,
        on_complete: Callable[[GenerationResponse, float | None], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_close: Callable[[GenerationResponse | None], None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._on_first_token = on_first_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_close = on_close

        self._start_time = time.perf_counter()
        self._first_token_time: float | None = None
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._response: GenerationResponse | None = None
        self._is_completed = False
        self._is_closed = False

    @property
    def response(self) -> GenerationResponse | None:
        """The finalized GenerationResponse if streaming has completed."""
        return self._response

    @property
    def time_to_first_token_ms(self) -> float | None:
        """Time in milliseconds from stream creation to first token."""
        if self._first_token_time is not None:
            return (self._first_token_time - self._start_time) * 1000
        return None

    def __iter__(self) -> Iterator[StreamChunk]:
        return self

    def __next__(self) -> StreamChunk:
        if self._is_closed:
            raise StopIteration

        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._finalize_on_complete()
            raise
        except Exception as err:
            if self._on_error and not self._is_completed:
                try:
                    self._on_error(err)
                except Exception:
                    pass
            raise

        if (chunk.text or chunk.thinking or chunk.tool_calls) and self._first_token_time is None:
            self._first_token_time = time.perf_counter()
            if self._on_first_token:
                try:
                    self._on_first_token(self.time_to_first_token_ms or 0.0)
                except Exception:
                    pass

        if chunk.text:
            self._text_parts.append(chunk.text)
        if chunk.thinking:
            self._thinking_parts.append(chunk.thinking)
        if chunk.tool_calls:
            self._tool_calls.extend(chunk.tool_calls)
        if chunk.response:
            self._response = chunk.response

        if chunk.is_finished:
            self._finalize_on_complete()

        return chunk

    def _finalize_on_complete(self) -> None:
        if self._is_completed:
            return
        self._is_completed = True

        if self._response is None:
            accumulated_text = "".join(self._text_parts)
            accumulated_thinking = "".join(self._thinking_parts) if self._thinking_parts else None
            self._response = GenerationResponse(
                text=accumulated_text,
                model="unknown",
                tool_calls=tuple(self._tool_calls),
                thinking=accumulated_thinking,
            )

        if self._on_complete:
            try:
                self._on_complete(self._response, self.time_to_first_token_ms)
            except Exception:
                pass

    def get_final_response(self) -> GenerationResponse:
        """Consume the remainder of the stream and return the final GenerationResponse."""
        if not self._is_completed:
            for _ in self:
                pass
        assert self._response is not None
        return self._response

    def close(self) -> None:
        """Close the stream and finalize resources."""
        if self._is_closed:
            return
        self._is_closed = True
        if not self._is_completed:
            self._finalize_on_complete()
        close_fn = getattr(self._iterator, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        if self._on_close:
            try:
                self._on_close(self._response)
            except Exception:
                pass

    def __enter__(self) -> StreamResponse:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class AsyncStreamResponse(AsyncIterator[StreamChunk]):
    """Asynchronous iterator and async context manager wrapping a streaming model generation."""

    def __init__(
        self,
        iterator: AsyncIterator[StreamChunk],
        *,
        on_first_token: Callable[[float], Any] | None = None,
        on_complete: Callable[[GenerationResponse, float | None], Any] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
        on_close: Callable[[GenerationResponse | None], Any] | None = None,
    ) -> None:
        self._iterator = iterator
        self._on_first_token = on_first_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_close = on_close

        self._start_time = time.perf_counter()
        self._first_token_time: float | None = None
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._response: GenerationResponse | None = None
        self._is_completed = False
        self._is_closed = False

    @property
    def response(self) -> GenerationResponse | None:
        """The finalized GenerationResponse if streaming has completed."""
        return self._response

    @property
    def time_to_first_token_ms(self) -> float | None:
        """Time in milliseconds from stream creation to first token."""
        if self._first_token_time is not None:
            return (self._first_token_time - self._start_time) * 1000
        return None

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self

    async def __anext__(self) -> StreamChunk:
        if self._is_closed:
            raise StopAsyncIteration

        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self._finalize_on_complete_async()
            raise
        except Exception as err:
            if self._on_error and not self._is_completed:
                try:
                    res = self._on_error(err)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    pass
            raise

        if (chunk.text or chunk.thinking or chunk.tool_calls) and self._first_token_time is None:
            self._first_token_time = time.perf_counter()
            if self._on_first_token:
                try:
                    res = self._on_first_token(self.time_to_first_token_ms or 0.0)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    pass

        if chunk.text:
            self._text_parts.append(chunk.text)
        if chunk.thinking:
            self._thinking_parts.append(chunk.thinking)
        if chunk.tool_calls:
            self._tool_calls.extend(chunk.tool_calls)
        if chunk.response:
            self._response = chunk.response

        if chunk.is_finished:
            await self._finalize_on_complete_async()

        return chunk

    async def _finalize_on_complete_async(self) -> None:
        if self._is_completed:
            return
        self._is_completed = True

        if self._response is None:
            accumulated_text = "".join(self._text_parts)
            accumulated_thinking = "".join(self._thinking_parts) if self._thinking_parts else None
            self._response = GenerationResponse(
                text=accumulated_text,
                model="unknown",
                tool_calls=tuple(self._tool_calls),
                thinking=accumulated_thinking,
            )

        if self._on_complete:
            try:
                res = self._on_complete(self._response, self.time_to_first_token_ms)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                pass

    async def get_final_response(self) -> GenerationResponse:
        """Consume the remainder of the stream and return the final GenerationResponse."""
        if not self._is_completed:
            async for _ in self:
                pass
        assert self._response is not None
        return self._response

    async def close(self) -> None:
        """Close the stream and finalize resources."""
        if self._is_closed:
            return
        self._is_closed = True
        if not self._is_completed:
            await self._finalize_on_complete_async()
        aclose_fn = getattr(self._iterator, "aclose", None)
        if callable(aclose_fn):
            try:
                await aclose_fn()
            except Exception:
                pass
        close_fn = getattr(self._iterator, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        if self._on_close:
            try:
                res = self._on_close(self._response)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                pass

    async def __aenter__(self) -> AsyncStreamResponse:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

