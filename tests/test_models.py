import pytest

from evidor import GenerationRequest, GenerationResponse, Message, Tool, ToolCall


def test_empty_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt must not be empty"):
        GenerationRequest(prompt="   ")

    with pytest.raises(ValueError, match="prompt must not be empty"):
        GenerationRequest(prompt=None)


def test_message_validation() -> None:
    msg = Message(role="user", content="hello", is_summary=False)
    assert msg.role == "user"
    assert msg.content == "hello"
    assert not msg.is_summary

    with pytest.raises(ValueError, match="message content must not be empty"):
        Message(role="assistant", content="")

    with pytest.raises(ValueError, match="message content must not be empty"):
        Message(role="system", content="   ")


def test_message_with_tool_calls_allows_empty_content() -> None:
    tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
    msg = Message(role="assistant", content="", tool_calls=(tc,))
    assert msg.content == ""
    assert msg.tool_calls == (tc,)

    # Tool role also allows empty or void content
    tool_msg = Message(role="tool", content="", tool_call_id="c1", name="search")
    assert tool_msg.role == "tool"
    assert tool_msg.content == ""


def test_tool_call_validation() -> None:
    tc = ToolCall(id="call_123", name="my_tool", arguments={"a": 1})
    assert tc.id == "call_123"
    assert tc.name == "my_tool"
    assert tc.arguments == {"a": 1}

    with pytest.raises(ValueError, match="tool call id must not be empty"):
        ToolCall(id="", name="my_tool", arguments={})

    with pytest.raises(ValueError, match="tool call name must not be empty"):
        ToolCall(id="call_1", name="", arguments={})


def test_generation_request_validation() -> None:
    msg = Message(role="user", content="hello")

    with pytest.raises(ValueError, match="provide either prompt or messages, not both"):
        GenerationRequest(prompt="hello", messages=[msg])

    with pytest.raises(ValueError, match="messages must not be empty"):
        GenerationRequest(messages=[])


def test_generation_request_with_tools() -> None:
    custom_tool = Tool(name="t", description="d", parameters={}, func=lambda: None)
    req = GenerationRequest(prompt="hello", tools=[custom_tool])
    assert len(req.tools) == 1
    assert req.tools[0].name == "t"


def test_generation_request_messages_and_prompt_alias() -> None:
    msg1 = Message(role="user", content="first")
    msg2 = Message(role="assistant", content="second")

    req = GenerationRequest(messages=[msg1, msg2])
    assert req.messages == (msg1, msg2)
    assert req.prompt == "second"


def test_generation_response() -> None:
    resp = GenerationResponse(text="output text", model="test-model")
    assert resp.text == "output text"
    assert resp.model == "test-model"
    assert resp.tool_calls == ()

    tc = ToolCall(id="call_1", name="fn", arguments={})
    resp_with_tools = GenerationResponse(text="", model="test-model", tool_calls=[tc])
    assert resp_with_tools.tool_calls == (tc,)
