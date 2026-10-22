"""Streaming event handler for Google Gemini."""

from collections.abc import Iterator
import json
from typing import Any

from evidor.models import GenerationResponse, StreamChunk
from ..base import ToolCallAccumulator
from .parser import GeminiResponseParser


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Safely extract a field from either an object attribute or dictionary key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class GeminiStreamHandler:
    """Processes Google GenAI raw stream events into normalized StreamChunk instances."""

    def __init__(self, parser: GeminiResponseParser) -> None:
        self.tool_acc = ToolCallAccumulator()
        self.step_id_map: dict[str, str] = {}
        self.thought_blocks: list[dict[str, Any]] = []
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.interaction_id: str | None = None
        self.parser = parser

    def process_event(self, event: Any) -> Iterator[StreamChunk]:
        """Process a single event from the Gemini raw stream."""
        ev_type = _get_field(event, "event_type") or _get_field(event, "type")
        idx = str(_get_field(event, "index", 0))

        if ev_type == "error":
            err_obj = _get_field(event, "error")
            err_msg = ""
            if isinstance(err_obj, dict):
                err_msg = err_obj.get("message") or str(err_obj)
            elif err_obj is not None:
                err_msg = _get_field(err_obj, "message") or str(err_obj)
            else:
                err_msg = str(event)
            raise RuntimeError(f"Gemini streaming error: {err_msg}")

        if ev_type in ("interaction.created", "interaction_created"):
            interaction = _get_field(event, "interaction")
            if interaction:
                int_id = _get_field(interaction, "id")
                if int_id and isinstance(int_id, str):
                    self.interaction_id = int_id

        delta_obj = _get_field(event, "delta")
        if delta_obj is not None and ev_type not in (
            "step.start",
            "step_start",
            "step.stop",
            "step_stop",
            "interaction.completed",
            "interaction_completed",
        ):
            delta_type = _get_field(delta_obj, "type")
            if delta_type == "text" or (
                (_get_field(delta_obj, "text") is not None) and (_get_field(delta_obj, "arguments") is None)
            ):
                txt = _get_field(delta_obj, "text", "") or ""
                if isinstance(txt, str) and txt:
                    self.text_parts.append(txt)
                    yield StreamChunk(text=txt, raw_event=event)
            elif delta_type in ("thought_summary", "thought"):
                th = ""
                content = _get_field(delta_obj, "content")
                if content is not None:
                    if isinstance(content, str):
                        th = content
                    elif isinstance(content, dict):
                        th = content.get("text", "") or ""
                    elif hasattr(content, "text"):
                        th = getattr(content, "text", "") or ""
                if not th:
                    th = _get_field(delta_obj, "text", "") or _get_field(delta_obj, "summary", "") or ""
                if isinstance(th, str) and th:
                    self.thinking_parts.append(th)
                    yield StreamChunk(thinking=th, raw_event=event)
            elif delta_type == "arguments_delta" or (_get_field(delta_obj, "arguments") is not None):
                # Prioritize self.step_id_map[idx] because event_id is an SSE stream token, not call_id
                call_id = (
                    self.step_id_map.get(idx)
                    or _get_field(event, "step_id")
                    or _get_field(event, "call_id")
                    or "call_1"
                )
                args_delta = _get_field(delta_obj, "arguments", "") or _get_field(delta_obj, "text", "") or ""
                if isinstance(args_delta, str) and args_delta:
                    self.tool_acc.add_arguments_delta(call_id, args_delta)
                elif isinstance(args_delta, dict):
                    self.tool_acc.add_arguments_delta(call_id, json.dumps(args_delta))
            elif delta_type in ("thought_signature", "thought"):
                sig = _get_field(delta_obj, "signature")
                if sig:
                    self.thought_blocks.append({
                        "type": "thought",
                        "thought": True,
                        "thought_signature": sig,
                    })

        if ev_type in ("step.start", "step_start"):
            step = _get_field(event, "step")
            if step and _get_field(step, "type") == "function_call":
                raw_call_id = _get_field(step, "id") or _get_field(event, "step_id")
                call_id = raw_call_id or f"call_{idx}"
                name = _get_field(step, "name", "") or ""
                self.step_id_map[idx] = call_id
                self.tool_acc.start_call(call_id, name)
                sig = _get_field(step, "signature")
                if sig:
                    self.thought_blocks.append({
                        "type": "function_call",
                        "id": call_id,
                        "name": name,
                        "thought_signature": sig,
                    })
            elif step and _get_field(step, "type") == "thought":
                self.step_id_map[idx] = f"thought_{idx}"
        elif ev_type in ("step.stop", "step_stop"):
            # Prioritize self.step_id_map[idx] because event_id is an SSE stream token, not call_id
            call_id = (
                self.step_id_map.pop(idx, None)
                or _get_field(event, "step_id")
                or _get_field(event, "call_id")
            )
            if call_id:
                tc = self.tool_acc.complete_call(call_id)
                if tc:
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)

        if ev_type in ("interaction.completed", "interaction_completed"):
            interaction = _get_field(event, "interaction") or event
            final_resp = self.parser.parse(interaction)
            accumulated_text = "".join(s for s in self.text_parts if isinstance(s, str))
            accumulated_thinking = "".join(s for s in self.thinking_parts if isinstance(s, str)) or None
            final_text = final_resp.text or accumulated_text
            final_thinking = final_resp.thinking or accumulated_thinking
            final_tools = final_resp.tool_calls or tuple(self.tool_acc._completed)
            resp_id = final_resp.id or self.interaction_id

            thought_blocks = list(self.thought_blocks)
            if accumulated_thinking and not any(tb.get("text") for tb in thought_blocks if isinstance(tb, dict)):
                thought_blocks.insert(0, {
                    "type": "thought",
                    "text": accumulated_thinking,
                    "thought": True,
                })

            final_resp = GenerationResponse(
                id=resp_id,
                text=final_text,
                model=final_resp.model,
                tool_calls=final_tools,
                usage=final_resp.usage,
                thinking=final_thinking,
                thought_blocks=tuple(thought_blocks) if thought_blocks else final_resp.thought_blocks,
                compaction=final_resp.compaction,
            )
            yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)
        elif hasattr(event, "text") and isinstance(event.text, str) and delta_obj is None:
            if event.text:
                self.text_parts.append(event.text)
                yield StreamChunk(text=event.text, raw_event=event)

    def finalize(self) -> Iterator[StreamChunk]:
        """Yield any in-flight accumulated tool calls at the end of the stream."""
        in_flight = []
        for call_id in list(self.tool_acc._calls.keys()):
            tc = self.tool_acc.complete_call(call_id)
            if tc is not None:
                in_flight.append(tc)
        if in_flight:
            yield StreamChunk(tool_calls=tuple(in_flight))


