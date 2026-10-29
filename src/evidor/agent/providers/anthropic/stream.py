"""Streaming event handler for Anthropic Claude."""

from collections.abc import Iterator
from typing import Any

from evidor.models import GenerationResponse, StreamChunk, ToolCall
from ..base import ToolCallAccumulator, build_token_usage
from .parser import AnthropicResponseParser, _get_field


class AnthropicStreamHandler:
    """Processes Anthropic raw stream events into normalized StreamChunk instances."""

    def __init__(self, model: str = "unknown", parser: AnthropicResponseParser | None = None) -> None:
        self.model = model
        self.parser = parser
        self.msg_id: str | None = None
        self.tool_acc = ToolCallAccumulator()
        self.block_id_map: dict[str, str] = {}
        self.block_thinking_map: dict[str, dict[str, Any]] = {}
        self.accumulated_text: list[str] = []
        self.text_thinking_parts: list[str] = []
        self.compaction_parts: list[str] = []
        self.completed_tool_calls: list[ToolCall] = []
        self.usage_dict: dict[str, int] = {}
        self.is_finished: bool = False

    def process_event(self, event: Any) -> Iterator[StreamChunk]:
        """Process a single event from the Anthropic raw stream."""
        raw_ev_type = _get_field(event, "type")
        ev_type = str(raw_ev_type) if raw_ev_type is not None else None

        if ev_type == "message_start":
            msg = _get_field(event, "message")
            if msg is not None:
                raw_id = _get_field(msg, "id")
                if isinstance(raw_id, str) and raw_id.strip():
                    self.msg_id = raw_id.strip()
                raw_model = _get_field(msg, "model")
                if isinstance(raw_model, str) and raw_model.strip():
                    self.model = raw_model.strip()
                raw_usage = _get_field(msg, "usage")
                if raw_usage is not None:
                    self._record_usage(raw_usage)

        elif ev_type == "content_block_start":
            idx = str(_get_field(event, "index", 0))
            cb = _get_field(event, "content_block")
            cb_type = _get_field(cb, "type") if cb is not None else None

            if cb_type == "tool_use":
                call_id = _get_field(cb, "id") or f"call_{idx}"
                name = _get_field(cb, "name") or "unknown_tool"
                self.block_id_map[idx] = str(call_id)
                self.tool_acc.start_call(str(call_id), str(name))
            elif cb_type == "thinking":
                th_text = _get_field(cb, "thinking") or ""
                self.block_thinking_map[idx] = {
                    "type": "thinking",
                    "parts": [str(th_text)] if th_text else [],
                    "signature": "",
                }
                if th_text:
                    self.text_thinking_parts.append(str(th_text))
                    yield StreamChunk(thinking=str(th_text), raw_event=event)
            elif cb_type == "redacted_thinking":
                data = _get_field(cb, "data") or ""
                self.block_thinking_map[idx] = {
                    "type": "redacted_thinking",
                    "data": str(data),
                }
            elif cb_type == "compaction":
                c = _get_field(cb, "content") or _get_field(cb, "text") or ""
                if c:
                    self.compaction_parts.append(str(c))

        elif ev_type == "content_block_delta":
            delta = _get_field(event, "delta")
            if delta is not None:
                idx = str(_get_field(event, "index", 0))
                delta_type = _get_field(delta, "type")

                if delta_type == "text_delta" or (hasattr(delta, "text") and not hasattr(delta, "thinking")):
                    txt = _get_field(delta, "text", "") or ""
                    if txt:
                        self.accumulated_text.append(str(txt))
                        yield StreamChunk(text=str(txt), raw_event=event)
                elif delta_type == "thinking_delta" or hasattr(delta, "thinking"):
                    th = _get_field(delta, "thinking", "") or ""
                    if th:
                        self.text_thinking_parts.append(str(th))
                        if idx in self.block_thinking_map and "parts" in self.block_thinking_map[idx]:
                            self.block_thinking_map[idx]["parts"].append(str(th))
                        else:
                            self.block_thinking_map[idx] = {
                                "type": "thinking",
                                "parts": [str(th)],
                                "signature": "",
                            }
                        yield StreamChunk(thinking=str(th), raw_event=event)
                elif delta_type == "signature_delta" or hasattr(delta, "signature"):
                    sig = _get_field(delta, "signature", "") or ""
                    if sig and idx in self.block_thinking_map:
                        self.block_thinking_map[idx]["signature"] = str(sig)
                elif delta_type == "input_json_delta" or hasattr(delta, "partial_json"):
                    call_id = self.block_id_map.get(idx, f"call_{idx}")
                    pj = _get_field(delta, "partial_json", "") or ""
                    self.tool_acc.add_arguments_delta(call_id, str(pj))
                elif delta_type == "compaction_delta":
                    c = _get_field(delta, "content", "") or ""
                    if c:
                        self.compaction_parts.append(str(c))

        elif ev_type == "content_block_stop":
            idx = str(_get_field(event, "index", 0))
            call_id = self.block_id_map.pop(idx, None)
            if call_id:
                tc = self.tool_acc.complete_call(call_id)
                if tc:
                    self.completed_tool_calls.append(tc)
                    yield StreamChunk(tool_calls=(tc,), raw_event=event)

        elif ev_type == "message_delta":
            raw_usage = _get_field(event, "usage")
            if raw_usage is not None:
                self._record_usage(raw_usage)

        elif ev_type == "message_stop":
            self.is_finished = True
            final_resp = self._build_final_response()
            yield StreamChunk(is_finished=True, response=final_resp, raw_event=event)

        elif ev_type in ("error", "api_error"):
            err_obj = _get_field(event, "error") or event
            err_msg = _get_field(err_obj, "message") or "Unknown Anthropic stream error"
            err_type = _get_field(err_obj, "type")
            full_msg = (
                f"Anthropic stream error ({err_type}): {err_msg}"
                if err_type
                else f"Anthropic stream error: {err_msg}"
            )
            raise RuntimeError(full_msg)

        elif hasattr(event, "text") and isinstance(event.text, str):
            if event.text:
                self.accumulated_text.append(event.text)
                yield StreamChunk(text=event.text, raw_event=event)

    def _record_usage(self, raw_usage: Any) -> None:
        """Accumulate token usage fields from message_start and message_delta."""
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "thinking_tokens",
            "reasoning_tokens",
        ):
            val = _get_field(raw_usage, field)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                self.usage_dict[field] = int(val)
        out_details = _get_field(raw_usage, "output_tokens_details")
        if out_details is not None:
            for field in ("thinking_tokens", "reasoning_tokens"):
                val = _get_field(out_details, field)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    self.usage_dict[field] = int(val)

    def _build_final_response(self) -> GenerationResponse:
        """Assemble the complete GenerationResponse from accumulated stream events."""
        thought_blocks: list[dict[str, Any]] = []
        for idx_key in sorted(self.block_thinking_map.keys(), key=lambda k: int(k) if k.isdigit() else 0):
            entry = self.block_thinking_map[idx_key]
            if entry.get("type") == "thinking":
                tb: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": "".join(entry.get("parts", [])),
                }
                sig = entry.get("signature")
                if sig:
                    tb["signature"] = sig
                thought_blocks.append(tb)
            elif entry.get("type") == "redacted_thinking":
                thought_blocks.append({
                    "type": "redacted_thinking",
                    "data": entry.get("data", ""),
                })

        text = "".join(self.accumulated_text)
        thinking_text = "".join(self.text_thinking_parts) if self.text_thinking_parts else None

        usage = None
        if self.parser is not None and self.usage_dict:
            usage = self.parser.parse_usage(self.usage_dict)
        elif self.usage_dict:
            it = self.usage_dict.get("input_tokens", 0)
            ot = self.usage_dict.get("output_tokens", 0)
            cr = self.usage_dict.get("cache_read_input_tokens", 0)
            cw = self.usage_dict.get("cache_creation_input_tokens", 0)
            rt = self.usage_dict.get("thinking_tokens") or self.usage_dict.get("reasoning_tokens") or 0
            usage = build_token_usage(
                prompt_tokens=it + cr,
                completion_tokens=ot,
                total_tokens=it + cr + ot,
                cached_tokens=cr,
                cache_creation_tokens=cw,
                reasoning_tokens=rt,
            )

        compaction = "".join(self.compaction_parts) if self.compaction_parts else None
        tool_calls = list(self.completed_tool_calls)

        return GenerationResponse(
            id=self.msg_id,
            text=text,
            model=self.model,
            tool_calls=tuple(tool_calls),
            usage=usage,
            thinking=thinking_text,
            thought_blocks=tuple(thought_blocks),
            compaction=compaction,
        )

    def finalize(self) -> Iterator[StreamChunk]:
        """Yield any in-flight accumulated tool calls and ensure stream termination."""
        in_flight: list[ToolCall] = []
        for call_id in list(self.tool_acc._calls.keys()):
            tc = self.tool_acc.complete_call(call_id)
            if tc is not None:
                in_flight.append(tc)
                self.completed_tool_calls.append(tc)

        if in_flight:
            yield StreamChunk(tool_calls=tuple(in_flight))

        if not self.is_finished:
            self.is_finished = True
            final_resp = self._build_final_response()
            yield StreamChunk(is_finished=True, response=final_resp)
