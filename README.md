# Evidor

Evidor is an open-source, provider-agnostic runtime for building AI agents. It provides a clean, unified interface for conversational agents across multiple model providers, with built-in context window management and automatic summarization.

## Install

Install the provider adapter(s) you need:

```bash
pip install "evidor[openai]"
# or: evidor[anthropic], evidor[gemini], evidor[mcp], evidor[otel], evidor[langfuse], evidor[phoenix], evidor[prometheus], evidor[all]
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

## Model Context Protocol (MCP)

Evidor includes first-class support for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Agents can connect to any MCP server (over local **stdio**, modern **Streamable HTTP**, or legacy **SSE**), automatically discover tools, execute them, and feed responses back into the model.

Install MCP support:

```bash
pip install "evidor[mcp]"
```

### Quickstart: Connecting an Agent to an MCP Server

You can pass `mcp_servers` directly when instantiating an `Agent`. For example, integrating an **Amazon MCP server** (e.g. AWS Bedrock, Amazon search, or a custom store MCP server):

```python
from evidor import Agent, OpenAIProvider

agent = Agent(
    OpenAIProvider(model="gpt-4.1-mini"),
    mcp_servers={
        "amazon": {
            "command": "uvx",
            "args": ["amazon-mcp-server"],
            "env": {"AWS_PROFILE": "default"},
        }
    },
)

# Tools exposed by the Amazon MCP server are automatically discovered and made available
response = agent.send("Search for Kindle Paperwhite on Amazon and give me a summary.")
print(response.text)

# Cleanly close subprocesses when done
agent.close()
```

Or using a context manager:

```python
with Agent(provider, mcp_servers={"amazon": {"command": "uvx", "args": ["amazon-mcp-server"]}}) as agent:
    response = agent.send("Find top noise-cancelling headphones on Amazon.")
    print(response.text)
```

### Async MCP with `send_async`

In asynchronous applications, use `send_async()` and `async with`:

```python
import asyncio
from evidor import Agent, AnthropicProvider

async def main():
    async with Agent(
        AnthropicProvider(model="claude-3-5-sonnet-20241022"),
        mcp_servers={
            "amazon": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-amazon"],
            }
        },
    ) as agent:
        response = await agent.send_async("Check the status of order ORD-12345.")
        print(response.text)

asyncio.run(main())
```

### Remote & Hosted MCP Servers (Streamable HTTP & SSE)

Evidor natively supports connecting to remote, hosted MCP servers. In modern MCP specifications, **Streamable HTTP** (`streamable-http` or `http`) is the primary standard for HTTP-based communication, while legacy **SSE** is supported for backwards compatibility:

```python
from evidor import Agent, OpenAIProvider, MCPServerConfig

# Modern Streamable HTTP (recommended for hosted MCP servers)
agent = Agent(
    OpenAIProvider("gpt-4.1-mini"),
    mcp_servers={
        "hosted-aws": {
            "url": "https://mcp.my-hosted-aws.com/mcp",
            "transport": "streamable-http",  # default for non-/sse URLs
            "headers": {"Authorization": "Bearer YOUR_API_KEY"},
        }
    },
)

# Or explicitly using factory methods:
config = MCPServerConfig.http(
    name="hosted-service",
    url="https://mcp.service.internal/mcp",
    headers={"x-api-key": "..."},
)
```

### Connecting Multiple MCP Servers

Connect multiple MCP servers simultaneously. Evidor automatically prevents tool naming collisions across servers:

```python
from evidor import Agent, GeminiProvider, MCPServerConfig

agent = Agent(
    GeminiProvider(model="gemini-2.5-flash"),
    mcp_servers=[
        # Stdio server
        MCPServerConfig.stdio(
            name="amazon",
            command="uvx",
            args=["amazon-mcp-server"],
        ),
        # Remote Streamable HTTP server
        MCPServerConfig.http(
            name="remote-inventory",
            url="https://inventory.internal/mcp",
            headers={"Authorization": "Bearer secret-token"},
        ),
        # Legacy SSE server
        MCPServerConfig.sse(
            name="weather",
            url="http://localhost:8080/sse",
        ),
    ],
)

response = agent.send("What is the weather in Seattle and recommend waterproof hiking boots on Amazon?")
print(response.text)
```

### `mcp_servers` vs `mcp_client`: When to Use Which

An `Agent` supports two parameters for MCP integration depending on whether the agent manages the connection lifecycle:

| Parameter | What You Pass | Best For |
| :--- | :--- | :--- |
| **`mcp_servers`** | Raw configuration (dicts, commands, URLs, or config files) | **Standard applications** where the agent owns the server processes. The agent automatically boots the client, connects to servers, and terminates subprocesses when closed. |
| **`mcp_client`** | An already-instantiated `MCPClient` instance | **Shared connections / multi-agent systems** (e.g. FastAPI backends) where a single client is kept open and shared across multiple agents without spawning duplicate subprocesses. |

#### Using `mcp_client` to Share Connections Across Agents

```python
from evidor import Agent, MCPClient, OpenAIProvider

# Boot a single shared client for your application
shared_mcp = MCPClient.from_file("claude_desktop_config.json")
shared_mcp.connect()

# Both agents reuse the exact same client without duplicating subprocesses:
researcher = Agent(OpenAIProvider("gpt-4.1-mini"), mcp_client=shared_mcp)
assistant = Agent(OpenAIProvider("gpt-4.1-mini"), mcp_client=shared_mcp)

response1 = researcher.send("Search records for client ABC.")
response2 = assistant.send("Verify credentials for client ABC.")

# Clean up once when the application terminates
shared_mcp.close()
```

### Advanced: Standalone `MCPClient` and Tool Composition

You can also use `MCPClient` standalone to inspect tools, retrieve resources, or mix MCP tools with standard Python tools:

```python
import asyncio
from evidor import Agent, MCPClient, OpenAIProvider, tool

@tool
def internal_calculator(x: int, y: int) -> int:
    return x * y

async def main():
    async with MCPClient({"amazon": {"command": "uvx", "args": ["amazon-mcp-server"]}}) as mcp:
        # Retrieve all Evidor Tool objects from the MCP server
        mcp_tools = await mcp.get_tools_async()

        # Combine MCP tools with local tools
        agent = Agent(OpenAIProvider("gpt-4.1-mini"), tools=[*mcp_tools, internal_calculator])
        response = await agent.send_async("Calculate total price for 3 units of item B08N5WRWNW.")
        print(response.text)

asyncio.run(main())
```

## Telemetry & Observability

Evidor includes a first-class, decoupled telemetry subsystem built on an asynchronous **actor-like producer-consumer runtime**. It captures distributed trace trees across agent turns, model generations, tool executions, and external MCP server invocations with zero overhead on the agent's critical path.

> **Zero Extra Dependencies for Custom Telemetry:**  
> Custom telemetry implementations and built-in sinks (`ConsoleSink`, `InMemorySink`) require **no external packages** (`pip install evidor`).  
> Optional adapters are available for industry-standard backends:
> * OpenTelemetry: `pip install "evidor[otel]"`
> * Langfuse: `pip install "evidor[langfuse]"`
> * Arize Phoenix: `pip install "evidor[phoenix]"`
> * Prometheus: `pip install "evidor[prometheus]"`

### Design Principles

- **Zero-Dependency Core**: Evidor core defines framework-neutral domain events and lifecycle operations. Telemetry adapters own protocol conversions and client exports.
- **Non-Blocking Emission**: Event emission on the agent thread takes under 1 microsecond via an in-memory bounded queue. Formatting, batching, and network transport happen on a dedicated background worker thread (`Evidor-Telemetry-Worker`).
- **Trace Context Propagation**: Automatically tracks parent-child span hierarchy across synchronous and asynchronous operations using `contextvars`.
- **Fault-Isolated**: Downstream observability backend timeouts, outages, or serialization errors never crash the agent.

---

### Built-in Sinks (Zero Dependencies)

Evidor includes built-in sinks that require zero external dependencies:

#### 1. `ConsoleSink`: Real-Time Developer Visibility
Stream structured events directly to `sys.stderr` or `sys.stdout`:

```python
from evidor import Agent, OpenAIProvider
from evidor.telemetry import ConsoleSink

sink = ConsoleSink()
agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=sink)
agent.send("What is the speed of light?")
```

#### 2. `InMemorySink`: Testing & Offline Assertions
Capture all events in memory to assert on token counts, latency, and tool invocations:

```python
from evidor import Agent, OpenAIProvider
from evidor.telemetry import InMemorySink, LLMCallEndEvent

sink = InMemorySink()
agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=sink)
agent.send("Hello agent!")

# Inspect recorded events
llm_events = sink.filter(LLMCallEndEvent)
print(f"Total tokens used: {llm_events[0].token_usage.total_tokens}")
```

---

### Observability Adapters

Evidor provides official, ultra-compact adapters for industry-leading observability platforms. Because Evidor's core runtime manages all context propagation, queuing, and background dispatch, each adapter requires **under ~100 lines of logic**:

| Adapter | Platform | Extra Install | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **`OpenTelemetrySink`** | OpenTelemetry, Datadog, Jaeger | `evidor[otel]` | Enterprise distributed tracing & OTel collectors |
| **`LangfuseSink`** | Langfuse (Cloud or self-hosted) | `evidor[langfuse]` | LLM engineering, prompt tracking & cost evaluation |
| **`PhoenixSink`** | Arize Phoenix | `evidor[phoenix]` | Local agent inspection with native OpenInference UI |
| **`PrometheusSink`** | Prometheus, Grafana | `evidor[prometheus]` | Operational counters, gauges & latency histograms |

#### 1. OpenTelemetry & OpenInference (`evidor[otel]`)

Convert domain events into distributed trace spans compliant with **OpenTelemetry GenAI** and **OpenInference** semantic conventions:

```bash
pip install "evidor[otel]"
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from evidor import Agent, OpenAIProvider, OpenTelemetrySink

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

otel_sink = OpenTelemetrySink(tracer_provider=provider)
agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=otel_sink)
response = agent.send("Search the documentation for vector embeddings.")
agent.close()
```

#### 2. Langfuse (`evidor[langfuse]`)

Trace agent executions, evaluate responses, monitor token usage, and track latency on the open-source [Langfuse](https://langfuse.com) platform:

```bash
pip install "evidor[langfuse]"
```

```python
from evidor import Agent, OpenAIProvider, LangfuseSink

# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from env
langfuse_sink = LangfuseSink()

agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=langfuse_sink)
response = agent.send("Summarize the quarterly earnings report.")
agent.close()  # Automatically flushes pending events and shuts down Langfuse client
```

#### 3. Arize Phoenix (`evidor[phoenix]`)

Send traces directly to [Arize Phoenix](https://phoenix.arize.com/) (local or cloud) formatted with native **OpenInference** metadata (`input.value`, `output.value`, `tool.parameters`, token breakdowns):

```bash
pip install "evidor[phoenix]"
```

```python
from evidor import Agent, OpenAIProvider, PhoenixSink

# Connects to Phoenix at http://localhost:6006/v1/traces by default
phoenix_sink = PhoenixSink(
    endpoint="http://localhost:6006/v1/traces",
    project_name="my-evidor-agent",
)

agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=phoenix_sink)
response = agent.send("Analyze customer sentiment from survey data.")
agent.close()
```

#### 4. Prometheus Metrics (`evidor[prometheus]`)

Export operational agent metrics (`evidor_agent_runs_total`, `evidor_llm_calls_total`, `evidor_tokens_total`, `evidor_tool_calls_total`, `evidor_errors_total`, and latency histograms) to Prometheus:

```bash
pip install "evidor[prometheus]"
```

```python
from evidor import Agent, OpenAIProvider, PrometheusSink

prom_sink = PrometheusSink()
agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=prom_sink)
agent.send("Calculate total revenue.")

# Export metrics exposition text (e.g., in a FastAPI /metrics endpoint):
metrics_text = prom_sink.export_text()
print(metrics_text)
```

---

#### Span Tree & Semantic Attributes

Evidor automatically builds the full execution tree:
```
agent.run                           [openinference.span.kind = "AGENT"]
  ├── llm.gpt-4.1-mini              [openinference.span.kind = "LLM"]
  └── tool.search_docs              [openinference.span.kind = "TOOL"]
        └── mcp.docs_server.search  [openinference.span.kind = "TOOL"]
```

Standard attributes automatically populated:
* `openinference.span.kind`: `AGENT`, `LLM`, or `TOOL`
* `input.value`, `output.value`, `tool.parameters`
* `gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.model`
* `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`
* `llm.input_prompt`, `llm.input_messages`, `llm.output_text`, `llm.output_tool_calls`
* `evidor.agent.run_id`, `evidor.tool.name`, `evidor.mcp.server_name`

---

### Advanced: Custom Sinks & Runtime Tuning

#### Implementing a Custom Sink
Implement the `TelemetrySink` protocol to send events to any custom database, webhook, or logging system:

```python
from collections.abc import Sequence
from evidor.telemetry import TelemetryEvent, TelemetrySink

class WebhookSink(TelemetrySink):
    def write(self, events: Sequence[TelemetryEvent]) -> None:
        for event in events:
            # Send batch payload over HTTP or write to database
            attrs = event.to_attributes()
            ...

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
```

#### Tuning the `TelemetryRuntime`
Configure queue capacity, batching thresholds, and overflow strategies:

```python
from evidor import Agent, OpenAIProvider
from evidor.telemetry import TelemetryRuntime, ConsoleSink

# Actor runtime with custom batching and bounded buffer
runtime = TelemetryRuntime(
    sinks=[ConsoleSink()],
    queue_size=2000,                # Max buffered events
    batch_size=100,                 # Batch size dispatched to sinks
    flush_interval_seconds=0.5,     # Max wait before flushing available events
    overflow_strategy="drop_oldest" # "drop_newest" (default) or "drop_oldest"
)

agent = Agent(OpenAIProvider("gpt-4.1-mini"), telemetry=runtime)
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

### Optional Extras Summary

| Extra | Purpose | Included Packages |
| :--- | :--- | :--- |
| `evidor[openai]` | OpenAI models (`gpt-4.1-mini`, `gpt-4o`) | `openai` |
| `evidor[anthropic]` | Anthropic Claude models (`claude-3-5-sonnet`) | `anthropic` |
| `evidor[gemini]` | Google Gemini models (`gemini-2.5-flash`) | `google-genai` |
| `evidor[mcp]` | Model Context Protocol servers | `mcp` |
| `evidor[otel]` | OpenTelemetry distributed tracing | `opentelemetry-api`, `opentelemetry-sdk` |
| `evidor[langfuse]` | Langfuse traces, generations & evaluation | `langfuse` |
| `evidor[phoenix]` | Arize Phoenix with OpenInference semantics | `arize-phoenix-otel`, `openinference-semantic-conventions` |
| `evidor[prometheus]` | Prometheus metrics and `/metrics` exposition | `prometheus-client` |
| `evidor[all]` | All model providers, MCP, and telemetry adapters | All optional extras |

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
