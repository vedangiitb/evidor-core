"""Streaming event handler for Anthropic Claude."""

from collections.abc import Iterator
from typing import Any

from evidor.models import StreamChunk
from ..base import ToolCallAccumulator


class AnthropicStreamHandler:
    """Processes Anthropic raw stream events into normalized StreamChunk instances."""

    def __init__(self) -> None:
        self.tool_acc = ToolCallAccumulator()
        self.block_id_map: dict[str, str] = {}

    def process_event(self, event: Any) -> Iterator[StreamChunk]:
        """Process a single event from the Anthropic raw stream."""
        ev_type = getattr(event, "type", None)

        if ev_type == "content_block_start":
            idx = str(getattr(event, "index", 0))
            cb = getattr(event, "content_block", None)
            if cb and getattr(cb, "type", None) == "tool_use":
                call_id = getattr(cb, "id", None) or f"call_{idx}"
                name = getattr(cb, "name", "") or ""
                self.block_id_map[idx] = call_id
                self.tool_acc.start_call(call_id, name)

        elif ev_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta is not None:
                delta_type = getattr(delta, "type", None)
                if delta_type == "text_delta" or hasattr(delta, "text"):
                    txt = getattr(delta, "text", "") or ""
                    if txt:
                        yield StreamChunk(text=txt, raw_event=event)
                elif delta_type == "thinking_delta" or hasattr(delta, "thinking"):
                    th = getattr(delta, "thinking", "") or ""
                    if th:
                        yield StreamChunk(thinking=th, raw_event=event)
                elif delta_type == "input_json_delta" or hasattr(delta, "partial_json"):
                    idx = str(getattr(event, "index", 0))
                    call_id = self.block_id_map.get(idx, f"call_{idx}")
                    pj = getattr(delta, "partial_json", "") or ""
                    self.tool_acc.add_arguments_delta(call_id, pj)

        elif ev_type == "content_block_stop":
            idx = str(getattr(event, "index", 0))
            call_id = self.block_id_map.pop(idx, None)
            if call_id:
                tc = self.tool_acc.complete_call(call_id)
                if tc:
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)

        elif ev_type == "message_stop":
            yield StreamChunk(is_finished=True, raw_event=event)
        elif hasattr(event, "text") and isinstance(event.text, str):
            if event.text:
                yield StreamChunk(text=event.text, raw_event=event)

    def finalize(self) -> Iterator[StreamChunk]:
        """Yield any in-flight accumulated tool calls at the end of the stream."""
        remaining_tools = self.tool_acc.finalize_all()
        if remaining_tools:
            yield StreamChunk(tool_calls=tuple(remaining_tools))

