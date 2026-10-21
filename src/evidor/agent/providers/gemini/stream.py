"""Streaming event handler for Google Gemini."""

from collections.abc import Iterator
from typing import Any

from evidor.models import GenerationResponse, StreamChunk
from ..base import ToolCallAccumulator
from .parser import GeminiResponseParser


class GeminiStreamHandler:
    """Processes Google GenAI raw stream events into normalized StreamChunk instances."""

    def __init__(self, parser: GeminiResponseParser) -> None:
        self.tool_acc = ToolCallAccumulator()
        self.step_id_map: dict[str, str] = {}
        self.thought_blocks: list[dict[str, Any]] = []
        self.text_parts: list[str] = []
        self.parser = parser

    def process_event(self, event: Any) -> Iterator[StreamChunk]:
        """Process a single event from the Gemini raw stream."""
        ev_type = getattr(event, "event_type", None) or getattr(event, "type", None)
        idx = str(getattr(event, "index", 0))

        delta_obj = getattr(event, "delta", None)
        if delta_obj is not None and ev_type not in (
            "step.start",
            "step_start",
            "step.stop",
            "step_stop",
            "interaction.completed",
            "interaction_completed",
        ):
            delta_type = getattr(delta_obj, "type", None)
            if delta_type == "text" or (hasattr(delta_obj, "text") and not hasattr(delta_obj, "arguments")):
                txt = getattr(delta_obj, "text", "") or ""
                if isinstance(txt, str) and txt:
                    self.text_parts.append(txt)
                    yield StreamChunk(text=txt, raw_event=event)
            elif delta_type in ("thought_summary", "thought"):
                th = getattr(delta_obj, "text", "") or getattr(delta_obj, "summary", "") or ""
                if isinstance(th, str) and th:
                    yield StreamChunk(thinking=th, raw_event=event)
            elif delta_type == "arguments_delta" or hasattr(delta_obj, "arguments"):
                call_id = (
                    getattr(event, "event_id", None)
                    or getattr(event, "step_id", None)
                    or self.step_id_map.get(idx)
                    or "call_1"
                )
                args_delta = getattr(delta_obj, "arguments", "") or getattr(delta_obj, "text", "") or ""
                if isinstance(args_delta, str) and args_delta:
                    self.tool_acc.add_arguments_delta(call_id, args_delta)
            elif delta_type in ("thought_signature", "thought"):
                sig = getattr(delta_obj, "signature", None)
                if isinstance(sig, str) and sig:
                    self.thought_blocks.append({
                        "type": "thought",
                        "thought": True,
                        "thought_signature": sig,
                    })

        if ev_type in ("step.start", "step_start"):
            step = getattr(event, "step", None)
            if step and getattr(step, "type", None) == "function_call":
                call_id = getattr(step, "id", None) or getattr(event, "event_id", None) or f"call_{idx}"
                name = getattr(step, "name", "") or ""
                self.step_id_map[idx] = call_id
                self.tool_acc.start_call(call_id, name)
                sig = getattr(step, "signature", None)
                if sig:
                    self.thought_blocks.append({
                        "type": "function_call",
                        "id": call_id,
                        "name": name,
                        "thought_signature": sig,
                    })
            elif step and getattr(step, "type", None) == "thought":
                self.step_id_map[idx] = f"thought_{idx}"
        elif ev_type in ("step.stop", "step_stop"):
            call_id = (
                getattr(event, "event_id", None)
                or getattr(event, "step_id", None)
                or self.step_id_map.pop(idx, None)
            )
            if call_id:
                tc = self.tool_acc.complete_call(call_id)
                if tc:
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)

        if ev_type in ("interaction.completed", "interaction_completed"):
            interaction = getattr(event, "interaction", event)
            final_resp = self.parser.parse(interaction)
            accumulated_text = "".join(s for s in self.text_parts if isinstance(s, str))
            final_text = final_resp.text or accumulated_text
            final_tools = final_resp.tool_calls or tuple(self.tool_acc._completed)
            final_resp = GenerationResponse(
                id=final_resp.id,
                text=final_text,
                model=final_resp.model,
                tool_calls=final_tools,
                usage=final_resp.usage,
                thinking=final_resp.thinking,
                thought_blocks=tuple(self.thought_blocks) if self.thought_blocks else final_resp.thought_blocks,
                compaction=final_resp.compaction,
            )
            yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)
        elif hasattr(event, "text") and isinstance(event.text, str) and delta_obj is None:
            if event.text:
                self.text_parts.append(event.text)
                yield StreamChunk(text=event.text, raw_event=event)

    def finalize(self) -> Iterator[StreamChunk]:
        """Yield any in-flight accumulated tool calls at the end of the stream."""
        remaining_in_flight = [
            tc
            for call_id in list(self.tool_acc._calls.keys())
            if (tc := self.tool_acc.complete_call(call_id)) is not None
        ]
        if remaining_in_flight:
            yield StreamChunk(tool_calls=tuple(remaining_in_flight))

