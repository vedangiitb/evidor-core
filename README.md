# Evidor

Evidor is a minimal, provider-agnostic LLM harness. It deliberately handles only
one thing: submit input and receive output.

## Install

Install the adapter(s) you need:

```bash
pip install "evidor[openai]"
# or: evidor[anthropic], evidor[gemini], evidor[all]
```

## Use

```python
from evidor import Agent, OpenAIProvider

agent = Agent(OpenAIProvider(model="gpt-4.1-mini"))
response = agent.run("Give a one-sentence explanation of dependency inversion.")
print(response.text)
```

Providers receive an API key explicitly or use their standard environment variable:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`.

The `Agent` depends only on the `ModelProvider` abstraction. To support another
provider, implement its single `generate(request)` method.
