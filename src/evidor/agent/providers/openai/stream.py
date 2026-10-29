"""Streaming event handler for OpenAI Responses API."""

from collections.abc import Iterator
from typing import Any

from evidor.models import StreamChunk, ToolCall
from ..base import ToolCallAccumulator
from .parser import OpenAIResponseParser, _get_field


class OpenAIStreamHandler:
    """Processes OpenAI Responses API stream events into normalized StreamChunk instances."""

    def __init__(self, parser: OpenAIResponseParser) -> None:
        self.tool_acc = ToolCallAccumulator()
        self.parser = parser
        self.item_id_map: dict[str, str] = {}

    def process_event(self, event: Any) -> Iterator[StreamChunk]:
        """Process a single event from the OpenAI Responses API raw stream."""
        raw_ev_type = _get_field(event, "type")
        ev_type = raw_ev_type if isinstance(raw_ev_type, str) else None

        if ev_type in ("response.output_text.delta", "response.text.delta"):
            delta = _get_field(event, "delta") or ""
            if delta:
                yield StreamChunk(text=str(delta), raw_event=event)
        elif ev_type in ("response.refusal.delta", "response.refusal_text.delta"):
            delta = _get_field(event, "delta") or ""
            if delta:
                yield StreamChunk(text=str(delta), raw_event=event)
        elif ev_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta = _get_field(event, "delta") or ""
            if delta:
                yield StreamChunk(thinking=str(delta), raw_event=event)
        elif ev_type == "response.output_item.added":
            item = _get_field(event, "item")
            item_type = _get_field(item, "type") if item else None
            if item and item_type in ("function_call", "custom_tool_call"):
                raw_call_id = _get_field(item, "call_id")
                raw_item_id = _get_field(item, "id")
                call_id = (
                    raw_call_id
                    if isinstance(raw_call_id, str) and raw_call_id.strip()
                    else (
                        raw_item_id
                        if isinstance(raw_item_id, str) and raw_item_id.strip()
                        else f"call_{len(self.item_id_map)+1}"
                    )
                )
                if isinstance(raw_item_id, str) and raw_item_id.strip():
                    self.item_id_map[raw_item_id.strip()] = call_id
                self.item_id_map[call_id] = call_id
                name = _get_field(item, "name") or ""
                self.tool_acc.start_call(call_id, str(name))
        elif ev_type in (
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input_delta",
        ):
            delta = _get_field(event, "delta") or ""
            raw_id = (
                _get_field(event, "item_id")
                or _get_field(event, "call_id")
                or _get_field(event, "id")
            )
            call_id = self.item_id_map.get(raw_id, raw_id) if raw_id else "call_1"
            if delta:
                self.tool_acc.add_arguments_delta(call_id, str(delta))
        elif ev_type in (
            "response.function_call_arguments.done",
            "response.custom_tool_call_input_done",
            "response.output_item.done",
        ):
            item = _get_field(event, "item")
            raw_id = (
                _get_field(item, "call_id")
                or _get_field(item, "id")
                or _get_field(event, "item_id")
                or _get_field(event, "call_id")
            )
            call_id = self.item_id_map.get(raw_id, raw_id) if raw_id else None
            if call_id and call_id in self.tool_acc._calls:
                event_args = _get_field(event, "arguments") or _get_field(item, "arguments")
                if event_args and not self.tool_acc._calls[call_id]["args_parts"]:
                    self.tool_acc.add_arguments_delta(call_id, str(event_args))
                item_name = _get_field(item, "name")
                if item_name and not self.tool_acc._calls[call_id]["name"]:
                    self.tool_acc._calls[call_id]["name"] = str(item_name)

                tc = self.tool_acc.complete_call(call_id)
                if tc:
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)
        elif ev_type in ("response.error", "error"):
            err_msg = _get_field(event, "message")
            if not err_msg:
                err_obj = _get_field(event, "error")
                err_msg = _get_field(err_obj, "message") if err_obj else None
            err_code = _get_field(event, "code")
            full_msg = (
                f"OpenAI stream error ({err_code}): {err_msg}"
                if err_code
                else f"OpenAI stream error: {err_msg or 'unknown error'}"
            )
            raise RuntimeError(full_msg)
        elif ev_type == "response.failed":
            resp = _get_field(event, "response")
            err_obj = _get_field(resp, "error") if resp else None
            err_msg = _get_field(err_obj, "message") if err_obj else None
            err_code = _get_field(err_obj, "code") if err_obj else None
            full_msg = (
                f"OpenAI response failed ({err_code}): {err_msg}"
                if err_code
                else f"OpenAI response failed: {err_msg or 'unknown failure'}"
            )
            raise RuntimeError(full_msg)
        elif ev_type == "response.completed":
            final_resp = self.parser.parse(_get_field(event, "response"))
            yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)

    def finalize(self) -> Iterator[StreamChunk]:
        """Yield any in-flight accumulated tool calls at the end of the stream."""
        in_flight: list[ToolCall] = []
        for call_id in list(self.tool_acc._calls.keys()):
            tc = self.tool_acc.complete_call(call_id)
            if tc is not None:
                in_flight.append(tc)
        if in_flight:
            yield StreamChunk(tool_calls=tuple(in_flight))
