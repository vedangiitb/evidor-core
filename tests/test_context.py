import pytest

from evidor.context import ConversationContext
from evidor.models import Message


def test_context_validation() -> None:
    with pytest.raises(ValueError, match="context_window must be at least 128 tokens"):
        ConversationContext(context_window=100, max_messages=10)

    with pytest.raises(ValueError, match="max_messages must be at least 3"):
        ConversationContext(context_window=1000, max_messages=2)


def test_compact_no_compaction_needed() -> None:
    ctx = ConversationContext(context_window=1000, max_messages=10)
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    result = ctx.compact(messages, lambda msgs: "summary")
    assert result == messages


def test_compact_summarizes_when_exceeding_max_messages() -> None:
    ctx = ConversationContext(context_window=1000, max_messages=3)
    system_msg = Message(role="system", content="system instruction")
    messages = [
        system_msg,
        Message(role="user", content="msg 1"),
        Message(role="assistant", content="reply 1"),
        Message(role="user", content="msg 2"),
    ]

    def mock_summarizer(msgs):
        return f"Summary of {len(msgs)} messages"

    result = ctx.compact(messages, mock_summarizer)

    assert result[0] == system_msg
    assert result[1].is_summary
    assert "Summary of 2 messages" in result[1].content
    assert result[2] == messages[-1]


def test_compact_handles_empty_summary_return() -> None:
    ctx = ConversationContext(context_window=1000, max_messages=3)
    messages = [
        Message(role="user", content="msg 1"),
        Message(role="assistant", content="reply 1"),
        Message(role="user", content="msg 2"),
        Message(role="assistant", content="reply 2"),
    ]

    result = ctx.compact(messages, lambda msgs: "   ")
    summary_msg = [m for m in result if m.is_summary][0]
    assert "Earlier conversation was omitted because no summary was returned." in summary_msg.content


def test_compact_retains_single_recent_when_over_token_budget() -> None:
    # Context window is small (128 tokens), single message is huge
    ctx = ConversationContext(context_window=128, max_messages=5)
    huge_message = Message(role="user", content="A" * 500)
    messages = [huge_message]

    result = ctx.compact(messages, lambda msgs: "summary")
    assert len(result) == 1
    assert result[0] == huge_message


def test_compact_merges_existing_summary() -> None:
    ctx = ConversationContext(context_window=1000, max_messages=3)
    old_summary = Message(role="system", content="Conversation summary:\nOld summary", is_summary=True)
    messages = [
        old_summary,
        Message(role="user", content="new msg 1"),
        Message(role="assistant", content="new reply 1"),
        Message(role="user", content="new msg 2"),
    ]

    summarized_received = []

    def mock_summarizer(msgs):
        summarized_received.extend(msgs)
        return "Updated summary"

    result = ctx.compact(messages, mock_summarizer)
    assert old_summary in summarized_received
    assert any(m.is_summary and "Updated summary" in m.content for m in result)
