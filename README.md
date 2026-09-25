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

## Build and release

Create distributable artifacts locally:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check dist/*
```

Releases are performed by GitHub Actions and use Conventional Commits to select
the next version:

- `fix: ...` creates a patch release.
- `feat: ...` creates a minor release.
- `feat!: ...` or a `BREAKING CHANGE:` footer creates a major release.

Pushes to `dev` publish `-dev.N` prereleases to TestPyPI. Pushes to `main`
publish stable releases to PyPI. Configure `TEST_PYPI_API_TOKEN` as a repository
or `dev` environment secret, and configure PyPI trusted publishing for the
`prod` environment before enabling the workflow.
