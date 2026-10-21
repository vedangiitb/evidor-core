# Evidor

Evidor is an open-source, provider-agnostic runtime for building AI agents. It provides a clean, modern interface for conversational agents across multiple model providers, with first-class support for stateful and stateless conversations, MCP, agent observability, tool integrations, and a wide range of built-in tools.

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

## Core Architecture: Execution Paradigms

Evidor cleanly separates AI execution into flexible paradigms:

1. **Stateful Conversations (`Agent`) (Default & Recommended)**: The primary and recommended way to build applications with Evidor. Conversations maintain state automatically across turns, so you don't have to pass the entire message history each time. Evidor manages session state, context continuation, and tool execution loops seamlessly under the hood.
2. **Stateless Agents (`Agent(..., stateless=True)`)**: When you want high-level agent capabilities (automatic multi-step tool execution loops, MCP tools, system prompts, and telemetry) but want every `.send()` / `.send_async()` call to execute completely independently without remembering past turns.
3. **Stateless LLM Calls (`provider.generate`)**: Direct single-turn or user-managed requests with tools, structured outputs, and raw message histories. Available when you need low-level control or want to manage message storage and persistence yourself.

---

## 1. Stateful Multi-Turn Conversations (Recommended)

Stateful conversations are the primary and recommended way to build agents in Evidor. With the `Agent` interface, conversations automatically remember context across turns—**you do not have to pass the entire message history each time**. Evidor handles session state, context tracking, and tool execution loops cleanly behind the scenes.

### Basic Multi-Turn Conversation

Use `send()` (or `send_async()`) to advance the conversation. Previous turns are remembered without re-passing the conversation history:

```python
from evidor import Agent, OpenAIProvider

agent = Agent(OpenAIProvider(model="gpt-6-astra"))

agent.send("My favorite color is navy blue.")
response = agent.send("What color did I say I like?")
print(response.text)
```

#### Streaming Responses

Pass `stream=True` to stream responses token-by-token across any provider:

```python
stream = agent.send("What color did I say I like?", stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)
```

### System Prompts

Configure an initial system prompt when creating an `Agent`. Evidor delivers the instructions natively across subsequent turns:

```python
from evidor import Agent, GeminiProvider

agent = Agent(
    GeminiProvider(model="gemini-3.8-flash"),
    system_prompt="You are a concise technical writer. Avoid buzzwords and respond in bullet points.",
)

response = agent.send("How does a TCP handshake work?")
print(response.text)
```

### How State and Context Are Handled Under the Hood

Under the hood, Evidor transparently maps stateful conversations to each model provider's optimal context mechanism, ensuring efficient token usage without requiring you to manually re-send past messages:

- **Server-Side State Chaining (OpenAI & Google Gemini)**:
  - **OpenAI (`OpenAIProvider`)**: Uses the Responses API `previous_response_id` (and optional `conversation_id`). State is maintained server-side; on each turn, Evidor transmits only the newest turn's delta messages.
  - **Google Gemini (`GeminiProvider`)**: Uses the Interactions API `previous_interaction_id`. State is maintained server-side; on each turn, Evidor transmits only the newest input step chained to the preceding interaction.
- **Ephemeral Prompt Caching (Anthropic Claude)**:
  - **Anthropic (`AnthropicProvider`)**: Claude's Messages API is stateless; `Agent` replays the message history locally on each turn (`"stateless_replay"`). Evidor automatically applies Anthropic's **Prompt Caching** (`cache_control={"type": "ephemeral"}`). The prompt prefix (system instructions, tools, and prior turns) is cached in KV-cache, reducing latency and cost by up to 90% without requiring server-side conversation storage.

Evidor exposes each provider's context management strategy through `agent.strategy`:

| Provider | Strategy (`agent.strategy`) | Context Architecture | Compaction Support | How It Works |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAIProvider** | `"response_chain"` | Server-Side Chaining (Responses API `previous_response_id`) | Native Server-Side (`compact_threshold`) | Only the new turn's delta messages / tool outputs are sent. When `compact_threshold` is configured, OpenAI executes server-side compaction automatically when tokens cross the threshold and preserves state across chained turns. |
| **GeminiProvider** | `"interaction_chain"` | Server-Side Chaining (Interactions API `previous_interaction_id`) | Managed in Interactions API | Only the new turn's delta messages are sent, linked by `previous_interaction_id`. Gemini maintains the conversation history on the server. |
| **AnthropicProvider** | `"stateless_replay"` | Client-Replayed History + Ephemeral Prompt Caching | Native Threshold Compaction (`compact_threshold`) | Claude's Messages API is stateless; `Agent` replays the message history on each turn. When `compact_threshold` is set, Claude generates a native `compaction` summary block, and Evidor replays the compaction block on subsequent turns so Claude automatically prunes prior turns. Combined with prompt caching (`enable_prompt_caching=True`), this delivers maximum context efficiency. |

### Native Provider Compaction (`compact_threshold`)

For long-running agents and multi-turn workflows, both OpenAI and Anthropic provide native compaction to summarize older context at a token threshold rather than hitting context window limits:

```python
from evidor import Agent, OpenAIProvider, AnthropicProvider

# 1. OpenAI: Automatic server-side compaction
openai_agent = Agent(
    OpenAIProvider(model="gpt-6-astra", compact_threshold=100000)
)

# 2. Anthropic Claude: Threshold compaction with optional custom instructions
claude_agent = Agent(
    AnthropicProvider(
        model="claude-3-7-sonnet-20250219",
        compact_threshold=150000,
        compact_instructions="Focus on preserving key architecture decisions and code snippets.",
    )
)

# You can also set compact_threshold directly on the Agent facade
agent = Agent(OpenAIProvider(model="gpt-6-astra"), compact_threshold=100000)
```

When compaction is triggered:
- **OpenAI**: Server-side compaction creates an opaque compaction item in the response stream. In Evidor's `response_chain` mode, passing `previous_response_id` carries this compaction forward seamlessly without manual pruning.
- **Anthropic**: Claude produces a `compaction` block at the start of the assistant response. Evidor preserves this compaction block and automatically formats it into the assistant turn of subsequent replayed requests. Claude automatically drops all history prior to the compaction block, keeping prompts compact and cache-optimized.

### Inspecting and Resetting State

You can inspect provider identifiers, strategy, message history, or reset the conversation at any time:

```python
agent = Agent(OpenAIProvider(model="gpt-6-astra"))
agent.send("Hello!")

# Inspect provider context strategy
print(agent.strategy)  # "response_chain"

# Inspect the latest provider response/interaction ID
print(agent.last_response_id)  # e.g. "resp_12345"

# Inspect the message history captured during this session
for msg in agent.messages:
    print(f"{msg.role}: {msg.content}")

# Reset the conversation state and provider response chain
agent.reset()  # or agent.clear_history()
print(agent.last_response_id)  # None
```

### Stateless Agent (`stateless=True`)

If you want the high-level capabilities of an `Agent`—automatic multi-step tool execution loops, MCP tools, system prompts, and telemetry—but want each `.send()` or `.send_async()` call to execute independently without remembering prior turns, pass `stateless=True`:

```python
from evidor import Agent, OpenAIProvider
from evidor.tools import tool

@tool
def get_stock_price(ticker: str) -> float:
    return 150.25

# Each send() call executes independently without cross-turn memory
agent = Agent(
    OpenAIProvider(model="gpt-6-astra"),
    system_prompt="You are a financial assistant.",
    tools=[get_stock_price],
    stateless=True,
)

# Turn 1: Invokes tool loop and returns answer
resp1 = agent.send("What is AAPL's price?")

# Turn 2: Starts completely fresh without prior turns or response chaining
resp2 = agent.send("What did I just ask?")
```

**Key Advantages of `stateless=True`**:
- **Automatic Inter-turn Isolation**: For OpenAI and Gemini, `previous_response_id` / `previous_interaction_id` are not chained across turns. For Anthropic, earlier turns are not replayed.
- **Intra-turn Tool Loops Intact**: Multi-step tool calls *within* a single turn (tool call $\to$ tool result $\to$ final response) work seamlessly because state is preserved during the turn.
- **Concurrency-Safe**: Each turn runs within an isolated ephemeral context frame, enabling concurrent `.send_async()` calls without cross-talk.

---

## 2. Stateless LLM Calls

While stateful conversations are recommended for most agent workflows, Evidor also supports direct stateless calls. In the stateless paradigm, you interact directly with the underlying `ModelProvider` using `GenerationRequest`. Evidor executes the call directly against modern provider endpoints without maintaining or mutating any hidden conversation state.

This is useful for one-off tasks, single-turn generations, or when your application already has its own persistence layer (e.g. database, chat storage, or custom state machine) and you want to pass raw message lists explicitly.

### Simple Requests

```python
from evidor import GenerationRequest, Message, OpenAIProvider

provider = OpenAIProvider(model="gpt-6-astra")

# Single-prompt shortcut
response = provider.generate(GenerationRequest(prompt="Explain Raft consensus in one paragraph."))
print(response.text)

# Multi-message request
request = GenerationRequest(
    messages=[
        Message(role="system", content="You are a distributed systems expert."),
        Message(role="user", content="What is quorum?"),
    ]
)
response = provider.generate(request)
print(response.text)
```

### User-Managed Message History

When your application handles its own persistence (e.g. database, chat storage, or custom state machine), you maintain the message list and pass it directly to `provider.generate()`. Unlike stateful `Agent` calls, in stateless mode you pass the full history on each request. Evidor never modifies, prunes, or summarises your messages behind your back:

```python
from evidor import GenerationRequest, Message, OpenAIProvider

provider = OpenAIProvider(model="gpt-6-astra")
chat_history: list[Message] = [
    Message(role="system", content="You are a concise coding assistant."),
]

# Turn 1
chat_history.append(Message(role="user", content="Recommend a fast key-value store in Go."))
response = provider.generate(GenerationRequest(messages=chat_history))
chat_history.append(Message(role="assistant", content=response.text))
print(f"Assistant: {response.text}")

# Turn 2
chat_history.append(Message(role="user", content="Does it support persistence?"))
response = provider.generate(GenerationRequest(messages=chat_history))
chat_history.append(Message(role="assistant", content=response.text))
print(f"Assistant: {response.text}")
```

### Stateless Tool Calling

You can supply tool definitions directly in a `GenerationRequest`. The model emits tool calls if needed, which you can execute statelessly in your own application loop:

```python
from evidor import GenerationRequest, Message, OpenAIProvider, Tool

tool_def = Tool(
    name="get_stock_price",
    description="Get the current price for a stock ticker.",
    parameters={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    func=lambda ticker: f"${ticker.upper()} is $150.00",
)

request = GenerationRequest(
    messages=[Message(role="user", content="What is the price of AAPL?")],
    tools=(tool_def,),
)
response = provider.generate(request)

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Model requested tool: {call.name} with args {call.arguments}")
```

---

## Streaming Responses

Evidor provides a provider-agnostic streaming interface across all supported providers (**OpenAI**, **Google Gemini**, and **Anthropic Claude**). By passing `stream=True`, you receive an iterable stream of lightweight chunks that emit tokens in real time while maintaining 100% compatibility with conversation memory, tool execution loops, and observability.

### Unified Developer Experience

The streaming interface is identical whether you are using OpenAI Responses API, Google GenAI Interactions API, or Anthropic Messages API:
- Pass `stream=True` to `agent.send()` or `await agent.send_async()`.
- Chunks stringify directly to their text content (`str(chunk)` returns `chunk.text`), so standard printing works immediately: `print(chunk, end="", flush=True)`.
- At the end of the stream, conversation history, compaction summaries, and server-side state (`last_response_id`) are automatically updated.

### Stateful Agent Streaming

#### Synchronous Streaming (`agent.send`)

```python
from evidor import Agent, OpenAIProvider

agent = Agent(OpenAIProvider(model="gpt-6-astra"))

# Stream responses token-by-token
stream = agent.send("Write a short haiku about debugging code.", stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)

# Access final accumulated response, token usage, and latency
print(f"\nModel: {stream.response.model}")
print(f"Total tokens: {stream.response.usage.total_tokens}")
print(f"Time to first token: {stream.time_to_first_token_ms:.1f}ms")
```

#### Asynchronous Streaming (`agent.send_async`)

In async environments (FastAPI, Starlette, async data workers), use `await agent.send_async(..., stream=True)`:

```python
import asyncio
from evidor import Agent, GeminiProvider

async def main():
    agent = Agent(GeminiProvider(model="gemini-3.8-flash"))
    stream = await agent.send_async("Explain how event loops work.", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)

    print(f"\nFinished with {stream.time_to_first_token_ms:.1f}ms TTFT")

asyncio.run(main())
```

---

### Streaming with Tools: Seamless Background Execution

When an agent needs to execute tools during a streaming turn:
1. **Clean Text Output**: Tool calling arguments stream and accumulate behind the scenes. `chunk.text` remains `""` so raw JSON arguments never pollute the user stream.
2. **Automatic Execution**: The agent detects and executes the tool function (concurrently via `asyncio.gather` if multiple tools were called) and updates the conversation history with the tool result.
3. **Follow-Up Stream**: The model automatically streams its final synthesized answer as text chunks to the caller.

#### Optional UI Indicators for Tool Calls

If you are building a user interface and want to show a loading badge or status indicator when a tool is called, you can inspect `chunk.tool_calls`:

```python
from evidor import Agent, GeminiProvider, tool

@tool
def check_order_status(order_id: str) -> str:
    """Look up order shipping status."""
    return f"Order {order_id} is out for delivery."

agent = Agent(GeminiProvider(model="gemini-3.8-flash"), tools=[check_order_status])
stream = agent.send("Where is my order ORD-992?", stream=True)

for chunk in stream:
    # chunk.tool_calls contains fully validated ToolCall instances when a tool is invoked
    if chunk.tool_calls:
        for call in chunk.tool_calls:
            print(f"\n[⚡ Executing tool: {call.name} with args {call.arguments}]")

    # Regular text tokens
    print(chunk, end="", flush=True)
```

---

### Streaming Thinking & Reasoning Tokens

For reasoning models (**OpenAI** `gpt-6-astra`, **Anthropic** `claude-3-7-sonnet-20250219`, or **Gemini** `gemini-3.8-flash` with thinking enabled), intermediate thought tokens are streamed via `chunk.thinking`:

```python
from evidor import Agent, AnthropicProvider

agent = Agent(AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=2048))
stream = agent.send("Solve: Which is greater, 9.11 or 9.9?", stream=True)

for chunk in stream:
    if chunk.thinking:
        print(f"\033[90m{chunk.thinking}\033[0m", end="", flush=True)  # Dim thinking text
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

---

### Stateless Provider Streaming

You can also stream direct single-turn LLM calls using `provider.generate()` or `provider.generate_async()`:

```python
from evidor import GenerationRequest, OpenAIProvider

provider = OpenAIProvider(model="gpt-6-astra")

request = GenerationRequest(
    prompt="Explain consensus in distributed systems.",
    stream=True,
)

# Synchronous streaming
stream = provider.generate(request)
for chunk in stream:
    print(chunk, end="", flush=True)

# Asynchronous streaming
# stream = await provider.generate_async(request)
# async for chunk in stream:
#     print(chunk, end="", flush=True)
```

---

### Context Managers & Resource Cleanup

Streams implement context manager protocols (`with` / `async with`) for deterministic cleanup if you break out of iteration early:

```python
# Sync context manager
with agent.send("List 5 facts about space.", stream=True) as stream:
    for chunk in stream:
        print(chunk, end="", flush=True)

# Async context manager
async with await agent.send_async("List 5 facts about space.", stream=True) as stream:
    async for chunk in stream:
        print(chunk, end="", flush=True)
```

If you need the final accumulated response without consuming the remaining chunks manually, call `stream.get_final_response()` (or `await stream.get_final_response()` in async mode).

---

### Telemetry & Time to First Token (TTFT)

When streaming is enabled, Evidor tracks streaming-specific metrics and lifecycle events:
- **`time_to_first_token_ms`**: Measured precisely from the moment the HTTP stream request starts until the first content or thinking token arrives. Accessible on `stream.time_to_first_token_ms` and recorded on `LLMCallEndEvent`.
- **`is_streaming`**: Boolean flag set to `True` on `LLMCallStartEvent` and `LLMCallEndEvent`.
- **Zero-Flooding Architecture**: Evidor's telemetry pipeline does not emit events per token. Lifecycle events are recorded at the call boundaries, preserving maximum streaming performance without saturating queues or telemetry sinks.

---

## Provider Retries & Capabilities

Evidor delegates retry handling directly to the underlying model provider SDKs (`openai`, `google-genai`, `anthropic`). Each provider SDK automatically handles transient network disruptions, connection resets, and rate limits using its native retry policies out of the box, avoiding redundant application-level retry wraps.

### Configurable Thinking & Reasoning Mode

Evidor provides first-class support for reasoning models across **OpenAI** (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`), **Anthropic** (`claude-3-7-sonnet-20250219`), and **Gemini** (`gemini-3.8-flash`, `gemini-3.8-live-extended-thinking`, `gemini-3.1-pro-preview`). Thinking can be configured either using provider-specific typed configuration classes or natural shorthands.

```python
from evidor import (
    Agent,
    AnthropicProvider,
    AnthropicThinkingConfig,
    GeminiProvider,
    GeminiThinkingConfig,
    OpenAIProvider,
    OpenAIThinkingConfig,
)

# 1. OpenAI: Configure reasoning effort ('low', 'medium', 'high', 'xhigh', 'max')
# Via shorthand:
provider = OpenAIProvider(model="gpt-6-astra", thinking="high")
# Or via typed config:
provider = OpenAIProvider(model="gpt-6-astra", thinking=OpenAIThinkingConfig(effort="high"))

# 2. Anthropic: Configure token budget or adaptive effort
# Shorthand int (sets budget_tokens >= 1024; auto-bumps max_tokens if needed):
provider = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=4096)
# Shorthand string (adaptive thinking):
provider = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking="high")
# Or via typed config:
provider = AnthropicProvider(
    model="claude-3-7-sonnet-20250219",
    thinking=AnthropicThinkingConfig(budget_tokens=4096),
)

# 3. Gemini: Configure thinking level ('minimal', 'low', 'medium', 'high') or budget
# Shorthand string (Gemini 3.x thinking_level):
provider = GeminiProvider(model="gemini-3.8-flash", thinking="high")
# Shorthand bool (False disables thinking; True enables dynamic thinking):
provider = GeminiProvider(model="gemini-3.8-flash", thinking=False)
# Or via typed config:
provider = GeminiProvider(
    model="gemini-3.8-flash",
    thinking=GeminiThinkingConfig(thinking_level="high", include_thoughts=True),
)
```

#### Inspecting Thinking Steps & Reasoning Tokens

When thinking is enabled and the provider outputs thought tokens, the response provides both the visible text and the internal reasoning scratchpad:

```python
agent = Agent(AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=2048))
response = agent.send("Solve this logic riddle step-by-step...")

# Final visible response
print(response.text)

# Model's internal reasoning scratchpad (if available)
if response.thinking:
    print(f"Reasoning Steps:\n{response.thinking}")

# Reasoning tokens recorded in usage metadata
if response.usage and response.usage.reasoning_tokens:
    print(f"Reasoning tokens consumed: {response.usage.reasoning_tokens}")
```

#### Multi-Turn Tool Calling & Cryptographic Signatures

Reasoning models from Anthropic and Gemini output thought blocks alongside cryptographic signatures (such as Anthropic's `signature` field or Gemini's thought parts). In multi-turn tool loops, these signatures must be preserved and replayed in subsequent conversation turns to avoid API verification errors. Evidor automatically captures and replays them in `agent.messages` and `thought_blocks` without requiring any manual handling.

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
    GeminiProvider(model="gemini-3.8-flash"),
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
    GeminiProvider(model="gemini-3.8-flash"),
    tools=[fetch_webpage],
    tool_timeout=10.0,      # Default tool timeout in seconds
    max_tool_iterations=10, # Max tool turns before forcing final synthesis
)
```

If a tool times out, raises an error, or if the model emits malformed JSON arguments, Evidor safely captures the error and feeds it back to the model as a `role="tool"` message so the LLM can self-correct.

### Native Asynchronous Conversations (`send_async`)

Evidor is built on an **async-first execution engine**. In async environments (like FastAPI, Starlette, Tornado, or asynchronous data pipelines), use `await agent.send_async(...)`. When called, all async tools execute natively on the caller's event loop without thread transitions, cleanly preserving shared connection pools, locks, and `contextvars`:

```python
import asyncio
from evidor import Agent, OpenAIProvider, tool

@tool
async def lookup_account(account_id: str) -> dict:
    return {"id": account_id, "status": "active"}

async def main() -> None:
    async with Agent(OpenAIProvider(model="gpt-4.1-mini"), tools=[lookup_account]) as agent:
        response = await agent.send_async("Check status for account 1234")
        print(response.text)

asyncio.run(main())
```

#### Key Async-First Features

- **Concurrent Multi-Tool Execution**: When an LLM emits multiple tool calls in a single turn, Evidor executes them **concurrently via `asyncio.gather()`**. This significantly reduces turn latency for complex agent workflows.
- **Native Async Provider Calls**: All bundled providers (`OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`) implement native `generate_async()` using modern asynchronous client libraries.
- **Persistent Connection Pooling (Sync & Async)**: Synchronous `agent.send()` transparently bridges calls through an encapsulated background daemon event loop. This preserves underlying HTTP/2 keep-alive connection pools across turns, eliminating TCP/TLS handshake latency on subsequent turns without tearing down sockets between calls.
- **Event Loop Guards (`RuntimeError`)**: Synchronous methods (`agent.send()`, `agent.close()`, `tool.execute()`, `mcp.get_tools()`, etc.) are protected with event loop guards. Calling them from inside an active event loop raises an explicit `RuntimeError` immediately rather than hanging the loop, directing you to use `await agent.send_async()`, `await agent.close_async()`, etc.
- **Clean Lifecycle & Resource Teardown**: Clean up agents, persistent background loops, and attached MCP sessions using `agent.close()`, `await agent.close_async()`, or context managers (`with Agent(...) as agent:` / `async with Agent(...) as agent:`).

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

### Webpage Reader

Use `read_webpage` (or `read_webpage_tool()`) to let agents fetch and read clean, token-bounded text from web pages, stripping HTML boilerplate automatically:

```python
from evidor import Agent, OpenAIProvider, read_webpage, read_webpage_tool

# Ready-to-use default tool (cleans HTML boilerplate, bounded to 15,000 chars)
agent = Agent(OpenAIProvider(model="gpt-4.1-mini"), tools=[read_webpage])

# Or customized with explicit character and timeout limits
custom_reader = read_webpage_tool(max_chars=8_000, timeout=10.0)
```

### Structured Evidence Store (`EvidenceStore`)

Prevent context pollution and hallucinations across multi-step research or analysis tasks by equipping agents with an in-memory structured evidence store:

```python
from evidor import Agent, EvidenceStore, GeminiProvider

store = EvidenceStore()

# store.tools provides record_evidence, list_evidence, and compare_evidence
agent = Agent(GeminiProvider(model="gemini-3.8-flash"), tools=[*store.tools])
agent.send("Analyze trade-offs between PostgreSQL and SQLite and record your findings.")

# Inspect structured findings in Python or format as Markdown
print(store.to_markdown())
for item in store.items:
    print(f"[{item.id}] {item.topic}: {item.finding}")
```

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
        AnthropicProvider(model="claude-3-7-sonnet-20250219"),
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
    GeminiProvider(model="gemini-3.8-flash"),
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

> [!TIP]
> **Matching Synchronous & Asynchronous Methods:**  
> `MCPClient` and `MCPSession` provide matching synchronous (`connect()`, `get_tools()`, `call_tool()`, `close()`) and asynchronous (`connect_async()`, `get_tools_async()`, `call_tool_async()`, `close_async()`) methods. Synchronous methods are protected with loop guards to prevent blocking the event loop; when called inside an event loop, they raise a `RuntimeError` prompting you to `await` the corresponding `_async` method.

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

Export operational agent metrics (`evidor_agent_runs_total`, `evidor_llm_calls_total`, `evidor_llm_retries_total`, `evidor_tokens_total`, `evidor_tool_calls_total`, `evidor_errors_total`, and latency histograms) to Prometheus:

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

#### Span Tree & Semantic Attributes

Evidor automatically builds the full execution tree across agent runs, LLM generations, and tool calls:
```
agent.run                     [openinference.span.kind = "AGENT"]
  ├── llm.gpt-4.1-mini        [openinference.span.kind = "LLM", status = "ok"]
  └── tool.search_docs        [openinference.span.kind = "TOOL"]
        └── mcp.docs_server.search  [openinference.span.kind = "TOOL"]
```

Standard attributes automatically populated:
* `openinference.span.kind`: `AGENT`, `LLM`, or `TOOL`
* `input.value`, `output.value`, `tool.parameters`
* `gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.model`
* `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`
* `llm.input_prompt`, `llm.input_messages`, `llm.output_text`, `llm.output_tool_calls`
* `llm.is_streaming`, `llm.time_to_first_token_ms` (when streaming)
* `evidor.agent.run_id`, `evidor.tool.name`, `evidor.mcp.server_name`


---

### Thinking Visibility in Telemetry (`capture_thinking`)

Internal thinking scratchpads can be lengthy and often contain confidential reasoning traces. All Evidor telemetry sinks default to **`capture_thinking=False`**, which redacts the raw thinking text while preserving reasoning token counts (`gen_ai.usage.reasoning_tokens` and Prometheus `evidor_tokens_total{token_type="reasoning"}`).

To make intermediate thinking steps visible in your traces, logs, or UI dashboards, enable `capture_thinking=True`:

```python
from evidor import Agent, AnthropicProvider
from evidor.telemetry import ConsoleSink, OpenTelemetrySink, PhoenixSink, LangfuseSink

# Enable thinking visibility on any sink
console_sink = ConsoleSink(capture_content=True, capture_thinking=True)
otel_sink = OpenTelemetrySink(capture_content=True, capture_thinking=True)
phoenix_sink = PhoenixSink(capture_content=True, capture_thinking=True)
langfuse_sink = LangfuseSink(capture_content=True, capture_thinking=True)

agent = Agent(
    AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=2048),
    telemetry=[console_sink, phoenix_sink],
)
response = agent.send("Analyze system security risks...")
agent.close()
```

#### Supported Telemetry Thinking Attributes & Semantics:
- **OpenTelemetry (`evidor[otel]`)**: Exports `gen_ai.reasoning.content` and `evidor.thinking.content` span attributes, plus `gen_ai.usage.reasoning_tokens`.
- **Arize Phoenix (`evidor[phoenix]`)**: Automatically outputs OpenInference reasoning blocks (`message_content.type = "reasoning"`) rendered directly in the Phoenix UI, and logs `llm.token_count.completion_details.reasoning`.
- **Langfuse (`evidor[langfuse]`)**: Attaches reasoning text to generation traces and populates `usage["reasoning"]`.
- **Prometheus (`evidor[prometheus]`)**: Tracks token consumption under `evidor_tokens_total{token_type="reasoning"}`.

---

### Privacy & PII Protection (Metadata-Only Mode)

In privacy-sensitive or regulated environments (such as healthcare, finance, or customer data processing), you may not want to persist raw user prompts, model outputs, or tool arguments to external observability backends, while still requiring distributed trace hierarchies, token usage, latency timings, and error analytics.

All Evidor telemetry sinks accept `capture_content: bool = True` and `capture_thinking: bool = False` parameters. Setting `capture_content=False` strips all raw text payloads before export:

```python
from evidor import Agent, OpenAIProvider
from evidor.telemetry import OpenTelemetrySink, LangfuseSink, PhoenixSink

# Sinks record span trees, token usage, and latencies, but redact all prompts, thinking, and outputs
otel_sink = OpenTelemetrySink(capture_content=False)
langfuse_sink = LangfuseSink(capture_content=False)
phoenix_sink = PhoenixSink(capture_content=False)

agent = Agent(
    OpenAIProvider("gpt-4.1-mini"),
    telemetry=[otel_sink, langfuse_sink],
)
response = agent.send("Sensitive patient record or financial data...")
agent.close()
```

| Field / Semantic Attribute | `capture_content=True`<br>`capture_thinking=False` (default) | `capture_content=True`<br>`capture_thinking=True` | `capture_content=False`<br>(metadata-only) |
| :--- | :--- | :--- | :--- |
| **Trace & Span IDs, Run IDs** | Preserved | Preserved | Preserved |
| **Duration & Timestamps** | Preserved | Preserved | Preserved |
| **Token Usage (incl. Reasoning Tokens)** | Preserved | Preserved | Preserved |
| **Model & Provider Names** | Preserved | Preserved | Preserved |
| **Tool & MCP Server Names** | Preserved | Preserved | Preserved |
| **Errors & Exceptions** | Preserved | Preserved | Preserved |
| **User Prompts & Messages** | Recorded | Recorded | **Omitted / Redacted** |
| **Model Output Text** | Recorded | Recorded | **Omitted / Redacted** |
| **Thinking / Reasoning Scratchpad** | **Omitted / Redacted** | **Recorded** | **Omitted / Redacted** |
| **Tool Arguments & Results** | Recorded | Recorded | **Omitted / Redacted** |

For custom sinks, the pure function `redact_event(event, capture_content=..., capture_thinking=...)` is exported directly from `evidor.telemetry`.

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

Supported providers, example models, and reasoning/thinking support:

| Provider | Example Models | Thinking Support | Environment Variable | Extra Install |
| --- | --- | --- | --- | --- |
| `OpenAIProvider` | `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, `gpt-4.1-mini` | `reasoning_effort` (`"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`) | `OPENAI_API_KEY` | `evidor[openai]` |
| `AnthropicProvider` | `claude-3-7-sonnet` | Extended budget (`budget_tokens >= 1024`) or adaptive | `ANTHROPIC_API_KEY` | `evidor[anthropic]` |
| `GeminiProvider` | `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview` | `thinking_level` (`"minimal"`, `"low"`, `"medium"`, `"high"`) or budget | `GEMINI_API_KEY` | `evidor[gemini]` |

API keys can be supplied explicitly via the `api_key` constructor argument or read automatically from environment variables. Custom base URLs for OpenAI-compatible providers (such as DeepSeek) can be supplied via the `base_url` constructor argument or `OPENAI_BASE_URL` environment variable:

```python
provider = OpenAIProvider(
    model="deepseek-flash",
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com",
)
```

### Modern Provider APIs & Zero Legacy Fallbacks

Evidor is in active beta and embraces breaking changes to maintain clean, modern integrations:
- **Google Gemini (`GeminiProvider`)**: Exclusively uses Google GenAI's modern Interactions API (`client.interactions.create`) from the official `google-genai` SDK (`from google import genai`). All legacy `generate_content` / `generate_content_stream` fallbacks and outdated candidate-parsing logic have been removed. Requires `google-genai>=1.0.0`.
- **OpenAI (`OpenAIProvider`)**: Exclusively uses OpenAI's modern Responses API (`client.responses.create`) with full support for `previous_response_id` chaining, reasoning models (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`), and modern tool schemas. Legacy Chat Completions methods have been removed. Requires `openai>=1.0.0`.
- **Anthropic (`AnthropicProvider`)**: Modern Messages API with native ephemeral prompt caching (`cache_control={"type": "ephemeral"}`) and adaptive/budget-based extended thinking.

### Custom Providers

`Agent` depends on the `ModelProvider` protocol. You can implement either synchronous `generate(request)`, native asynchronous `generate_async(request)`, or both:

- **Async-First**: If you implement `generate_async(request)`, the synchronous `generate(request)` is automatically derived for you via `run_async_in_sync()`.
- **Sync Fallback**: If you only implement `generate(request)`, `generate_async(request)` automatically offloads execution using `asyncio.to_thread()` without blocking the caller's event loop.

```python
from evidor import GenerationRequest, GenerationResponse, ModelProvider

class CustomAsyncProvider(ModelProvider):
    def __init__(self, model: str = "custom-model") -> None:
        self.model = model

    def with_model(self, model: str) -> "CustomAsyncProvider":
        return CustomAsyncProvider(model=model)

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        # Asynchronously call your custom model backend
        prompt_text = request.prompt
        return GenerationResponse(text="Custom async response", model=self.model)
```

### Optional Extras Summary

| Extra | Purpose | Included Packages |
| :--- | :--- | :--- |
| `evidor[openai]` | OpenAI models (`gpt-6-astra`, `gpt-4.1-mini`) | `openai` |
| `evidor[anthropic]` | Anthropic Claude models (`claude-3-7-sonnet`) | `anthropic` |
| `evidor[gemini]` | Google Gemini models (`gemini-3.8-flash`, `gemini-3.5-flash-lite`) | `google-genai` |
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

Pushes to `dev` publish `-dev.N` prereleases to PyPI. Pushes to `main` publish stable releases to PyPI. Configure PyPI trusted publishing for both the `prod` and `dev` environments before enabling the workflow.

