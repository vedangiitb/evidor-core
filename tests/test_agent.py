from evidor import Agent, GenerationRequest, GenerationResponse, ModelProvider


class FakeProvider:
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(text=f"Received: {request.prompt}", model="fake")


def test_agent_delegates_to_its_provider() -> None:
    agent = Agent(FakeProvider())

    response = agent.run("hello")

    assert response == GenerationResponse(text="Received: hello", model="fake")
    assert isinstance(FakeProvider(), ModelProvider)
