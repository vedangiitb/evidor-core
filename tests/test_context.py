import pytest

from evidor.agent.context.context import ConversationContext
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
    ctx = ConversationContext(context_window=1000, max_messages=3, chain_milestones=False)
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


def test_context_validation_ratios() -> None:
    with pytest.raises(ValueError, match="low_water_ratio must be less than high_water_ratio"):
        ConversationContext(context_window=1000, max_messages=10, low_water_ratio=0.85, high_water_ratio=0.5)


def test_compact_chains_milestones_without_resummarizing_old_milestones() -> None:
    # chain_milestones defaults to True
    ctx = ConversationContext(context_window=1000, max_messages=3)
    assert ctx.chain_milestones is True
    old_summary = Message(role="system", content="Conversation summary:\nMilestone 1", is_summary=True)
    messages = [
        old_summary,
        Message(role="user", content="turn 1"),
        Message(role="assistant", content="turn 1 reply"),
        Message(role="user", content="turn 2"),
    ]

    summarized_received = []

    def mock_summarizer(msgs):
        summarized_received.extend(msgs)
        return "Milestone 2"

    result = ctx.compact(messages, mock_summarizer)

    # Old summary must NOT be re-summarized (preserves prefix cache!)
    assert old_summary not in summarized_received
    # Both milestones are preserved in order
    summary_messages = [m for m in result if m.is_summary]
    assert len(summary_messages) == 2
    assert summary_messages[0] == old_summary
    assert "Milestone 2" in summary_messages[1].content
    assert result[-1] == messages[-1]


def test_compact_hysteresis_batch_pruning() -> None:
    # High water: 10 messages, Low water: 4 messages (0.4 ratio)
    ctx = ConversationContext(
        context_window=10000,
        max_messages=10,
        high_water_ratio=0.85,
        low_water_ratio=0.4,
        chain_milestones=True,
    )
    # Start with 11 messages (triggers compaction)
    messages = [
        Message(role="system", content="sys instruction"),
        *[Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}") for i in range(10)],
    ]
    assert len(messages) == 11

    result = ctx.compact(messages, lambda msgs: "Batch summary")

    # Pruned down to low-water mark (system + 1 summary + remaining recent messages <= 4)
    assert result[0] == messages[0]
    assert result[1].is_summary
    assert len(result) <= 5  # System + Summary + <= 3 recent

    # Adding another turn does NOT trigger compaction again (hysteresis buffer!)
    next_messages = [*result, Message(role="user", content="next turn")]
    next_result = ctx.compact(next_messages, lambda msgs: "Should not be called")
    assert next_result == next_messages


def test_summarizer_excludes_permanent_system_prompt() -> None:
    from unittest.mock import MagicMock
    from evidor.agent.context.summarization import ConversationSummarizer

    mock_provider = MagicMock()
    summarizer = ConversationSummarizer(provider=mock_provider)

    permanent_system = Message(role="system", content="Permanent developer instructions: adhere strictly to PEP8")
    existing_summary = Message(role="system", content="Conversation summary:\nPrevious work", is_summary=True)
    user_msg = Message(role="user", content="Calculate 42")
    assistant_msg = Message(role="assistant", content="The answer is 42")

    messages = [permanent_system, existing_summary, user_msg, assistant_msg]
    req = summarizer._build_request(messages)

    user_payload = req.messages[1].content
    # Permanent system prompt must NOT be in the summarizer's transcript
    assert "Permanent developer instructions" not in user_payload
    # Conversational messages and previous summary should be in the transcript
    assert "[PREVIOUS SUMMARY OF EARLIER CONVERSATION]:\nPrevious work" in user_payload
    assert "USER: Calculate 42" in user_payload
    assert "ASSISTANT: The answer is 42" in user_payload

