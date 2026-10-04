# Evidor

Evidor is an open-source, provider-agnostic runtime for building AI agents. It provides a clean, unified interface for conversational agents across multiple model providers, with built-in context window management and automatic summarization.

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

## Tools & Function Calling

Evidor provides first-class primitives for defining tools and equipping agents with automatic function-calling execution loops.

### Defining Tools with `@tool`

Transform any Python function into a provider-neutral tool using the `@tool` decorator. Evidor automatically derives the JSON Schema for parameters from type annotations and extracts descriptions from docstrings (Google, Sphinx, or plain styles):

```python
from evidor import tool

@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather forecast for a given location.

    Args:
        location: City and country or state, e.g. 'San Francisco, CA'.
        unit: Temperature scale ('celsius' or 'fahrenheit').
    """
    return f"Weather in {location}: 22° {unit}, clear skies."

@tool(name="calc_add", description="Add two numbers together.")
def add(a: float, b: float) -> float:
    return a + b
```

### Equipping Agents with Tools

Pass your tools directly into `Agent(..., tools=[...])`. When a model decides to call one or more tools, the agent automatically executes them, feeds the results back to the model as `role="tool"` messages, and loops until the model generates a final response:

```python
from evidor import Agent, GeminiProvider, tool

@tool
def search_database(query: str) -> list[str]:
    """Search internal records by keyword."""
    return [f"Record 1 matching '{query}'", f"Record 2 matching '{query}'"]

agent = Agent(
    GeminiProvider(model="gemini-2.5-flash"),
    tools=[search_database],
    max_tool_iterations=10,  # Maximum tool execution turns per send() (default: 10)
)

response = agent.send("Can you check our records for 'alpha project'?")
print(response.text)
```

### Async Tools & Execution Timeouts

Evidor seamlessly supports asynchronous functions and execution timeouts:

```python
import asyncio
from evidor import Agent, GeminiProvider, tool

# Async functions are automatically executed without blocking or crashing
@tool(timeout=5.0)  # Timeout in seconds
async def fetch_webpage(url: str) -> str:
    """Fetch content from a URL."""
    await asyncio.sleep(0.1)
    return f"Contents of {url}"

# Configure an agent with a default timeout for all its tools
agent = Agent(
    GeminiProvider(model="gemini-2.5-flash"),
    tools=[fetch_webpage],
    tool_timeout=10.0,      # Default tool timeout in seconds
    max_tool_iterations=10, # Max tool turns before forcing final synthesis
)
```

If a tool times out, raises an error, or if the model emits malformed JSON arguments, Evidor safely captures the error and feeds it back to the model as a `role="tool"` message so the LLM can self-correct.

### Native Asynchronous Conversations (`send_async`)

In async environments (like FastAPI, Tornado, or asynchronous data pipelines), use `await agent.send_async(...)`. When called, all async tools execute natively on the caller's event loop without thread transitions, cleanly preserving shared connection pools, locks, and `contextvars`:

```python
import asyncio
from evidor import Agent, OpenAIProvider, tool

@tool
async def lookup_account(account_id: str) -> dict:
    return {"id": account_id, "status": "active"}

async def main() -> None:
    agent = Agent(OpenAIProvider(model="gpt-4.1-mini"), tools=[lookup_account])
    response = await agent.send_async("Check status for account 1234")
    print(response.text)

asyncio.run(main())
```

### Manual Tool Definition

For dynamic or programmatic tools without a standard Python function signature, instantiate `Tool` directly:

```python
from evidor import Tool

custom_tool = Tool(
    name="query_sql",
    description="Run a read-only SQL query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL statement"}
        },
        "required": ["query"],
    },
    func=lambda query: f"Results for: {query}",
    timeout=5.0,
)
```

Evidor's tool calling works consistently across **OpenAI**, **Anthropic**, and **Gemini**, adapting each provider's underlying tool schema and message format transparently.

### Built-in Tools

Evidor includes a couple of dependency-free tools you can opt into. They are not attached to agents automatically:

```python
from evidor import Agent, OpenAIProvider, calculator, get_current_time

agent = Agent(
    OpenAIProvider(model="gpt-4.1-mini"),
    tools=[calculator, get_current_time],
)
```

`get_current_time` returns an ISO 8601 timestamp and accepts an IANA timezone name (UTC by default). `calculator` evaluates basic arithmetic without executing Python code.

Create filesystem tools scoped to an existing directory for listing, searching, reading, creating, updating, and deleting files. Paths supplied by the model are resolved under that root, and file reads and writes and result counts have configurable limits:

```python
from evidor import Agent, OpenAIProvider, calculator, filesystem_tools, get_current_time

agent = Agent(
    OpenAIProvider(model="gpt-4.1-mini"),
    tools=[
        calculator,
        get_current_time,
        *filesystem_tools("./project", max_file_bytes=100_000, max_results=100),
    ],
)
```

`filesystem_tools` returns six tools:

- `list_files` lists files in the root or a subdirectory.
- `read_file` reads a UTF-8 text file.
- `search_files` searches UTF-8 text files for a literal string.
- `create_file` creates a new UTF-8 text file; it fails if the path already exists. Its parent directory must already exist.
- `write_file` replaces the contents of an existing UTF-8 text file.
- `delete_file` deletes an existing file (directories cannot be deleted).

All paths stay within the configured root. `max_file_bytes` limits the size of a file read or written (including initial content passed to `create_file`); `max_results` bounds listing and search results, and `max_files_scanned` bounds recursive search work.

### Web Search

Use `web_search()` to create a provider-neutral agent tool. The tool depends only on the `WebSearchProvider` interface, so the selected search service can be changed without changing the agent setup. The bundled Tavily, Exa, and Brave adapters use only Python's standard library; set the provider's API key environment variable (or pass `api_key=` explicitly).

```bash
export TAVILY_API_KEY="..."
# or EXA_API_KEY="..." / BRAVE_SEARCH_API_KEY="..."
```

```python
from evidor import Agent, OpenAIProvider, TavilySearchProvider, web_search

search = web_search(TavilySearchProvider(), max_results=5)
agent = Agent(OpenAIProvider(model="gpt-4.1-mini"), tools=[search])

response = agent.send("Find the current Python release notes and cite the sources.")
print(response.text)
```

Switch providers by replacing `TavilySearchProvider()` with `ExaSearchProvider()` or `BraveSearchProvider()`. To use another service, implement the small `WebSearchProvider` contract: `search(query: str, *, max_results: int) -> list[SearchResult]`; both types are exported from `evidor`.

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
