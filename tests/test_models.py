import pytest

from evidor import GenerationRequest


def test_empty_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt must not be empty"):
        GenerationRequest(prompt="   ")
