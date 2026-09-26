# Evidor

Evidor is a minimal, provider-agnostic LLM harness. It provides a clean, unified interface for conversational agents across multiple model providers, with built-in context window management and automatic summarization.

## Install

Install the provider adapter(s) you need:

```bash
pip install "evidor[openai]"
# or: evidor[anthropic], evidor[gemini], evidor[all]
```

## Quickstart

```python
from evidor import Agent, OpenAIProvider

agent = Agent(OpenAIProvider(model="gpt-4.1-mini"))
response = agent.send("Give a one-sentence explanation of dependency inversion.")
print(response.text)
```

## Core Usage

### Multi-turn Conversations

The `Agent` maintains session state across conversation turns. Use `send()` to advance the conversation:

```python
from evidor import Agent, AnthropicProvider

agent = Agent(AnthropicProvider(model="claude-3-5-sonnet-20241022"))

agent.send("My favorite color is navy blue.")
response = agent.send("What color did I say I like?")
print(response.text)
```

### System Prompts

You can configure an initial system prompt when creating an `Agent`. The system prompt is preserved across conversation compaction and history resets:

```python
from evidor import Agent, GeminiProvider

agent = Agent(
    GeminiProvider(model="gemini-2.5-flash"),
    system_prompt="You are a concise technical writer. Avoid buzzwords and respond in bullet points.",
)

response = agent.send("How does a TCP handshake work?")
print(response.text)
```

### Context Window Management & Compaction

`Agent` automatically manages message history to stay within token and message limits using `ConversationContext`:

```python
from evidor import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_MESSAGES, Agent, OpenAIProvider

agent = Agent(
    OpenAIProvider(model="gpt-4.1-mini"),
    context_window=16_000,  # Max token budget (default: 16,000)
    max_messages=50,        # Max messages retained before compaction (default: 50)
)
```

When conversation history exceeds `max_messages` or the estimated token budget:
- Older messages are automatically summarized using the configured model provider.
- The summary is injected as a bounded summary message (`Message(role="system", ..., is_summary=True)`).
- The permanent system prompt and recent conversational turns remain intact.

### Configurable Summarization Model

By default, the agent uses its primary provider and model for summarization during compaction. You can configure a specific model name string or a separate `ModelProvider` instance (e.g. to use a faster, lighter model for background summaries):

```python
from evidor import Agent, GeminiProvider, OpenAIProvider

# Option 1: Configure a different model name on the same provider
agent = Agent(
    OpenAIProvider(model="gpt-4.1"),
    summarization_model="gpt-4.1-mini",
)

# Option 2: Provide an entirely separate ModelProvider instance
agent = Agent(
    OpenAIProvider(model="gpt-4.1"),
    summarization_model=GeminiProvider(model="gemini-2.5-flash"),
)
```

### Inspecting and Clearing History

You can inspect the retained conversation history or reset turns at any time:

```python
# Inspect current history (tuple of Message objects)
for message in agent.messages:
    prefix = "[SUMMARY] " if message.is_summary else ""
    print(f"{prefix}{message.role}: {message.content}")

# Clear conversation turns while keeping the original system prompt
agent.clear_history()
```

### Low-level Generation with `GenerationRequest`

For direct provider calls without stateful session management, instantiate messages and requests directly:

```python
from evidor import GenerationRequest, Message, OpenAIProvider

provider = OpenAIProvider(model="gpt-4.1-mini")

# Multi-message request
request = GenerationRequest(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain Raft consensus in one paragraph."),
    ]
)
response = provider.generate(request)
print(response.text)

# Single-prompt compatibility shortcut
simple_request = GenerationRequest(prompt="Hello!")
response = provider.generate(simple_request)
```

## Providers

Supported providers and their typical models:

| Provider | Example Model | Environment Variable | Extra Install |
| --- | --- | --- | --- |
| `OpenAIProvider` | `gpt-4.1-mini` | `OPENAI_API_KEY` | `evidor[openai]` |
| `AnthropicProvider` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` | `evidor[anthropic]` |
| `GeminiProvider` | `gemini-2.5-flash` | `GEMINI_API_KEY` | `evidor[gemini]` |

API keys can be supplied explicitly via the `api_key` constructor argument or read automatically from environment variables.

### Custom Providers

`Agent` depends only on the `ModelProvider` protocol. To support another LLM or backend, implement the `generate(request)` method:

```python
from evidor import GenerationRequest, GenerationResponse, ModelProvider

class CustomProvider:
    def __init__(self, model: str = "custom-model") -> None:
        self.model = model

    def with_model(self, model: str) -> "CustomProvider":
        return CustomProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt_text = request.prompt
        # or iterate over request.messages
        return GenerationResponse(text="Custom response", model=self.model)
```

## Build and release

Create distributable artifacts locally:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check dist/*
```

Releases are performed by GitHub Actions and use Conventional Commits to select the next version:

- `fix: ...` creates a patch release.
- `feat: ...` creates a minor release.
- `feat!: ...` or a `BREAKING CHANGE:` footer creates a major release.

Pushes to `dev` publish `-dev.N` prereleases to TestPyPI. Pushes to `main` publish stable releases to PyPI. Configure `TEST_PYPI_API_TOKEN` as a repository or `dev` environment secret, and configure PyPI trusted publishing for the `prod` environment before enabling the workflow.
