# AI Agent Development Guidelines & Provider Documentation Rules

This project integrates with multiple AI providers (Gemini, OpenAI, Anthropic, etc.). AI models suffer from training cutoffs and frequently generate code referencing obsolete SDKs, deprecated parameters, or outdated models (such as `gpt-4o`, `o1`, `o3-mini`, `gemini-2.5-flash`, etc.).

**Whenever writing, refactoring, or reviewing AI provider integrations, you MUST consult the active MCP documentation servers or authoritative live docs before producing code.**

---

## 1. Provider MCP Documentation Lookup Protocol

1. **Check Live Docs First**:
   - For **Gemini API**: Query the `gemini-docs` MCP tools (`gemini_search_docs`, `gemini_get_doc`).
   - For **OpenAI API**: Query the `openaiDeveloperDocs` MCP tools (`search_openai_docs`, `fetch_openai_doc`, `list_openai_docs`, `list_api_endpoints`, `get_openapi_spec`).
2. **Fallback**: If MCP tools are offline or unavailable in the current context, query the live docs via web search or fetch endpoints:
   - Gemini: `https://ai.google.dev/gemini-api/docs/llms.txt`
   - OpenAI: `https://developers.openai.com/llms.txt` or `https://platform.openai.com/docs`
3. **Never Guess API Signatures or Models**: Verify SDK parameter names, return types, schema serialization, and model identifiers against the live documentation.

---

## 2. Google Gemini API Guidelines (Gemini 3.x)

- **Consult MCP Tools**: Always search `gemini-docs` (`gemini_search_docs` / `gemini_get_doc`) or refer to `https://ai.google.dev/gemini-api/docs/llms.txt` prior to implementing Gemini features.
- **SDK Requirements**:
  - **Python**: Use `google-genai` (`from google import genai`, `from google.genai import types`).
  - **TypeScript / JavaScript**: Use `@google/genai`.
  - **STRICTLY PROHIBITED**: Never use legacy packages (`google-generativeai` or `@google/generative-ai`).
- **Client Instantiation & Calls**:
  - Initialize using `client = genai.Client(...)`.
  - Standard generation: `client.models.generate_content(...)` or streaming with `client.models.generate_content_stream(...)`.
- **Thinking Configuration in Gemini 3.x**:
  - Use `thinking_level` string enum (`"minimal"`, `"low"`, `"medium"`, `"high"`):
    ```python
    types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="medium")
    )
    ```
  - **Avoid**: Numeric `thinking_budget` is no longer recommended across Gemini 3.x models.
- **Sampling Parameters**:
  - Do **NOT** set `temperature`, `top_p`, or `top_k` for Gemini 3.x models; they are calibrated and optimized for default reasoning settings.
- **Function Calling Strict Matching**:
  - Every `FunctionResponse` MUST include the `id` from the corresponding `FunctionCall`.
  - The `name` in the response must match the `name` in the call.
  - Return exactly one `FunctionResponse` for each `FunctionCall` received.
- **Active Recommended Models**:
  - `gemini-3.8-flash`: Flagship Flash model for long-horizon software engineering, autonomous agents, and enterprise workflows.
  - `gemini-3.5-flash-lite`: Fastest, most cost-effective model for high-throughput execution.
  - `gemini-3.1-pro-preview`: Advanced intelligence and complex problem solving.
  - `gemini-3.8-live` & `gemini-3.8-live-extended-thinking`: For real-time voice and multimodal agents.
  - **OUTDATED / RESTRICTED**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-1.5-*`. Do not use for new development.

---

## 3. OpenAI API Guidelines (GPT-6 Era)

- **Consult MCP Tools**: Query `openaiDeveloperDocs` (`search_openai_docs`, `fetch_openai_doc`, `get_openapi_spec`) before writing OpenAI client code.
- **SDK Requirements**:
  - **Python**: Use `openai` v1.x/v3.x+ (`from openai import OpenAI, AsyncOpenAI`).
  - **TypeScript / JavaScript**: Use `openai` v4.x+.
  - **STRICTLY PROHIBITED**: Pre-1.0 legacy syntax (`import openai; openai.ChatCompletion.create(...)` or global `openai.api_key = ...`).
- **Client Instantiation & API Usage**:
  - Always instantiate a client: `client = OpenAI(...)` or `client = AsyncOpenAI(...)`.
  - **Responses API is Standard**: Use `client.responses.create(...)` for tool calling and reasoning models. For latest reasoning models, Chat Completions does not support tool calling.
  - Function calling tools in Responses API:
    ```python
    tools=[{"type": "function", "name": "...", "description": "...", "parameters": {...}}]
    ```
  - Tool responses in Responses API:
    ```python
    {"type": "function_call_output", "call_id": "...", "output": "..."}
    ```
- **Reasoning Configuration**:
  - Use `reasoning={"effort": "..."}` with supported levels: `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`.
- **Structured JSON Outputs**:
  - Use Pydantic models via `client.responses.create` or `client.beta.chat.completions.parse(...)`.
- **Active Recommended Models**:
  - `gpt-6-astra`: Flagship reasoning model for complex coding, reasoning, computer use, and agents.
  - `gpt-6.1-sol`: Near-Astra performance at lower cost for complex coding.
  - `gpt-6-luna`: Cost-sensitive, high-volume focused reasoning workloads.
  - `gpt-4.1`, `gpt-4.1-mini`: Smartest non-reasoning models.
  - **OUTDATED / DEPRECATED**: `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, `gpt-4.5-preview`, `gpt-3.5-turbo`. Do not use for new development.

---

## 4. General Provider Guidelines

- **Environment Secrets**: Never hardcode API keys. Always read from environment variables (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or dedicated configuration objects.
- **Error Handling**: Use provider-specific exceptions from the modern SDKs (e.g., `from google.genai.errors import APIError`, `openai.APIError`).
- **Telemetry & Token Usage**: Parse both standard completions tokens and Responses API token details (`cached_tokens` and `reasoning_tokens`).
