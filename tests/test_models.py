import pytest

from evidor import GenerationRequest, GenerationResponse, Message


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


def test_generation_request_validation() -> None:
    msg = Message(role="user", content="hello")

    with pytest.raises(ValueError, match="provide either prompt or messages, not both"):
        GenerationRequest(prompt="hello", messages=[msg])

    with pytest.raises(ValueError, match="messages must not be empty"):
        GenerationRequest(messages=[])


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
