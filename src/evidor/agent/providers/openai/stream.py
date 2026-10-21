"""Streaming event handler for OpenAI Responses API."""

from collections.abc import Iterator
from typing import Any

from evidor.models import StreamChunk
from ..base import ToolCallAccumulator
from .parser import OpenAIResponseParser


class OpenAIStreamHandler:
    """Processes OpenAI Responses API stream events into normalized StreamChunk instances."""

    def __init__(self, parser: OpenAIResponseParser) -> None:
        self.tool_acc = ToolCallAccumulator()
        self.parser = parser

    def process_event(self, event: Any) -> Iterator[StreamChunk]:
        """Process a single event from the OpenAI Responses API raw stream."""
        ev_type = getattr(event, "type", None)
        if ev_type in ("response.output_text.delta", "response.text.delta"):
            delta = getattr(event, "delta", "") or ""
            if delta:
                yield StreamChunk(text=delta, raw_event=event)
        elif ev_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta = getattr(event, "delta", "") or ""
            if delta:
                yield StreamChunk(thinking=delta, raw_event=event)
        elif ev_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or "call_1"
                name = getattr(item, "name", "") or ""
                self.tool_acc.start_call(call_id, name)
        elif ev_type == "response.function_call_arguments.delta":
            delta = getattr(event, "delta", "") or ""
            call_id = getattr(event, "item_id", None) or getattr(event, "call_id", None) or "call_1"
            self.tool_acc.add_arguments_delta(call_id, delta)
        elif ev_type in ("response.function_call_arguments.done", "response.output_item.done"):
            item = getattr(event, "item", None)
            call_id = getattr(event, "item_id", None) or getattr(item, "call_id", None) if item else None
            if not call_id and item:
                call_id = getattr(item, "id", None)
            if call_id:
                tc = self.tool_acc.complete_call(call_id)
                if tc:
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)
        elif ev_type == "response.completed":
            final_resp = self.parser.parse(getattr(event, "response", None))
            yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)

    def finalize(self) -> Iterator[StreamChunk]:
        """Yield any in-flight accumulated tool calls at the end of the stream."""
        remaining_tools = self.tool_acc.finalize_all()
        if remaining_tools:
            yield StreamChunk(tool_calls=tuple(remaining_tools))

