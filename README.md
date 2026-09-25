# Evidor

Evidor is a minimal, provider-agnostic LLM harness. It deliberately handles only
one thing: submit input and receive output.

> This is an early alpha release. The public API currently covers synchronous,
> single-prompt generation only.

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

## Build and publish

Create distributable artifacts locally:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check dist/*
```

To release, first create a PyPI project named `evidor` (or select an available
distribution name), then upload with a PyPI API token:

```bash
python -m twine upload dist/*
```

Use the `__token__` username and store the token outside this repository (for
example, in your shell's secure environment variables or CI secret store).
