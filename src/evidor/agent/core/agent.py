"""The public conversational agent and session facade."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import threading
from typing import Any, Literal, overload

from evidor.models import (
    AsyncStreamResponse,
    GenerationResponse,
    Message,
    StreamResponse,
)
from evidor.telemetry import TelemetryRuntime, TelemetrySink
from evidor.tools import Tool
from evidor.tools.core.tool_executor import ToolExecutor
from evidor.utils import BackgroundEventLoop, check_sync_allowed, run_async_in_sync
from .context import ConversationContext
from .runner import TurnExecutor, sync_stream_bridge
from .telemetry_bridge import AgentTelemetryBridge
from ..providers.base import ModelProvider


class Agent:
    """A conversational session coordinating a provider, context state, tools, and telemetry.

    Serves as the high-level Facade composing modular sub-components:
    - ConversationContext: Encapsulates message history and provider context chaining.
    - ToolExecutor: Manages local tools and executes calls with timeouts.
    - AgentTelemetryBridge: Bridges trace spans and metrics across runs and LLM calls.
    - TurnExecutor: Executes multi-turn loops, tool iterations, and streaming chunks.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        mcp_servers: Any = None,
        mcp_client: Any = None,
        max_tool_iterations: int = 10,
        tool_timeout: float | None = None,
        system_prompt: str | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        store: bool | None = None,
        compact_threshold: int | None = None,
        telemetry: TelemetryRuntime | TelemetrySink | Sequence[TelemetrySink] | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._compact_threshold = compact_threshold or getattr(provider, "compact_threshold", None)
        if (
            compact_threshold is not None
            and hasattr(provider, "_compact_threshold")
            and getattr(provider, "_compact_threshold") is None
        ):
            provider._compact_threshold = compact_threshold

        self._max_tool_iterations = max_tool_iterations
        self._tool_timeout = tool_timeout

        # MCP client initialization
        if mcp_client is not None:
            self._mcp_client = mcp_client
        elif mcp_servers is not None:
            from evidor.mcp import MCPClient

            self._mcp_client = MCPClient(mcp_servers)
        else:
            self._mcp_client = None

        # Tool execution setup
        self._user_tools = list(tools) if tools is not None else []
        self._tool_executor = ToolExecutor(self._user_tools, tool_timeout)
        self._mcp_tools_synced = False

        # Modular sub-components
        self._context = ConversationContext(
            strategy=self.strategy,
            system_prompt=system_prompt,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            store=store,
            compact_threshold=self._compact_threshold,
        )
        self._telemetry_bridge = AgentTelemetryBridge(telemetry)
        self._runner = TurnExecutor(
            provider=self._provider,
            context=self._context,
            tool_executor=self._tool_executor,
            telemetry_bridge=self._telemetry_bridge,
            max_tool_iterations=max_tool_iterations,
        )

        # Background event loop and thread synchronization
        self._background_loop: BackgroundEventLoop | None = None
        self._lock = threading.RLock()

    # --------------------------------------------------------------------------
    # Public Properties
    # --------------------------------------------------------------------------

    @property
    def provider(self) -> ModelProvider:
        """The underlying model provider."""
        return self._provider

    @property
    def strategy(self) -> str:
        """The context management strategy employed by this provider:
        - 'response_chain': OpenAI Responses API server-side chaining
        - 'interaction_chain': Gemini Interactions API server-side chaining
        - 'stateless_replay': Replays message history (used by Anthropic Claude and stateless models)
        """
        return getattr(self._provider, "context_strategy", "stateless_replay")

    @property
    def last_response_id(self) -> str | None:
        """The identifier of the last response or interaction in this conversation."""
        return self._context.last_response_id

    @property
    def conversation_id(self) -> str | None:
        """The conversation identifier, if attached to an external conversation resource."""
        return self._context.conversation_id

    @property
    def system_prompt(self) -> str | None:
        """The optional system instructions configured for this agent."""
        return self._system_prompt

    @property
    def compact_threshold(self) -> int | None:
        """The configured token threshold for provider-native compaction, if any."""
        return self._compact_threshold

    @property
    def telemetry(self) -> TelemetryRuntime | None:
        """The TelemetryRuntime configured on this agent, if any."""
        return self._telemetry_bridge.telemetry

    @property
    def messages(self) -> tuple[Message, ...]:
        """The recorded conversation history."""
        return self._context.messages

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The tools configured on this agent."""
        self._sync_mcp_tools()
        return self._tool_executor.tools

    @property
    def mcp_client(self) -> Any:
        """The MCPClient associated with this agent, if configured."""
        return self._mcp_client

    @property
    def tool_timeout(self) -> float | None:
        """Default per-tool execution timeout in seconds, if configured."""
        return self._tool_timeout

    @property
    def _sync_loop(self) -> BackgroundEventLoop:
        """The persistent background event loop for synchronous agent calls."""
        if self._background_loop is None:
            self._background_loop = BackgroundEventLoop(name=f"AgentLoop-{id(self)}")
        return self._background_loop

    # --------------------------------------------------------------------------
    # Conversation State Management
    # --------------------------------------------------------------------------

    def clear_history(self) -> None:
        """Clear conversation history and reset provider state identifiers."""
        self._context.clear()

    def reset(self) -> None:
        """Alias for clear_history."""
        self.clear_history()

    # --------------------------------------------------------------------------
    # MCP Synchronization
    # --------------------------------------------------------------------------

    def _sync_mcp_tools(self, force: bool = False) -> None:
        """Synchronize tools from any configured MCP client into the executor."""
        if self._mcp_client is None or (self._mcp_tools_synced and not force):
            return
        self._sync_loop.run(
            lambda: self._sync_mcp_tools_async(force=force),
            error_message=(
                "Cannot synchronize MCP tools synchronously from a running event loop. "
                "Access agent.tools outside an event loop or run an async agent turn first."
            ),
        )

    async def _sync_mcp_tools_async(self, force: bool = False) -> None:
        """Asynchronously synchronize tools from any configured MCP client into the executor."""
        if self._mcp_client is None or (self._mcp_tools_synced and not force):
            return
        mcp_tools = await self._mcp_client.get_tools_async()
        self._tool_executor = ToolExecutor([*self._user_tools, *mcp_tools], self._tool_timeout)
        self._runner.update_tool_executor(self._tool_executor)
        self._mcp_tools_synced = True

    # --------------------------------------------------------------------------
    # Public Execution APIs
    # --------------------------------------------------------------------------

    @overload
    def send(self, message: str, *, stream: Literal[False] = False) -> GenerationResponse: ...

    @overload
    def send(self, message: str, *, stream: Literal[True]) -> StreamResponse: ...

    @overload
    def send(
        self, message: str, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse: ...

    def send(
        self, message: str, *, stream: bool = False
    ) -> GenerationResponse | StreamResponse:
        """Advance one user turn synchronously."""
        if not stream:
            with self._lock:
                return self._sync_loop.run(
                    lambda: self.send_async(message, stream=False),
                    error_message=(
                        "Agent.send() cannot be called from a running event loop. "
                        "Use 'await agent.send_async()' instead."
                    ),
                )

        check_sync_allowed(
            "Agent.send(..., stream=True) cannot be called from a running event loop. "
            "Use 'await agent.send_async(..., stream=True)' instead."
        )
        with self._lock:
            bridge = sync_stream_bridge(
                self._sync_loop, lambda: self.send_async(message, stream=True)
            )
            return StreamResponse(bridge)

    @overload
    async def send_async(
        self, message: str, *, stream: Literal[False] = False
    ) -> GenerationResponse: ...

    @overload
    async def send_async(
        self, message: str, *, stream: Literal[True]
    ) -> AsyncStreamResponse: ...

    @overload
    async def send_async(
        self, message: str, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse: ...

    async def send_async(
        self, message: str, *, stream: bool = False
    ) -> GenerationResponse | AsyncStreamResponse:
        """Asynchronously advance one user turn, coordinating provider-native context and tool calls."""
        await self._sync_mcp_tools_async()
        if stream:
            return AsyncStreamResponse(self._runner.stream_turn_async(message))

        return await self._runner.run_turn_async(message)

    # --------------------------------------------------------------------------
    # Lifecycle & Cleanup
    # --------------------------------------------------------------------------

    def close(self) -> None:
        """Close any associated MCP client connections, background event loop, and flush telemetry."""
        with self._lock:
            try:
                if self._background_loop is not None and self._background_loop.is_running:
                    self._background_loop.run(
                        self._close_async_resources,
                        error_message=(
                            "Agent.close() cannot be called from a running event loop. "
                            "Use 'await agent.close_async()' or 'async with agent:' instead."
                        ),
                    )
                else:
                    running_loop = None
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass

                    if running_loop is not None:
                        if self._mcp_client is not None:
                            check_sync_allowed(
                                "Agent.close() cannot be called from a running event loop. "
                                "Use 'await agent.close_async()' or 'async with agent:' instead."
                            )
                        running_loop.create_task(self._close_async_resources())
                    else:
                        run_async_in_sync(
                            self._close_async_resources,
                            error_message=(
                                "Agent.close() cannot be called from a running event loop. "
                                "Use 'await agent.close_async()' or 'async with agent:' instead."
                            ),
                        )
                self._close_sync_resources()
            finally:
                self._telemetry_bridge.flush_and_close()
                if self._background_loop is not None:
                    self._background_loop.close()
                    self._background_loop = None

    async def close_async(self) -> None:
        """Asynchronously close any associated MCP client connections, background event loop, and flush telemetry."""
        try:
            if self._background_loop is not None and self._background_loop.is_running:
                bg_loop = self._background_loop.loop
                current_loop = None
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if bg_loop is not None and bg_loop != current_loop:
                    fut = asyncio.run_coroutine_threadsafe(self._close_async_resources(), bg_loop)
                    await asyncio.wrap_future(fut)
                else:
                    await self._close_async_resources()
            else:
                await self._close_async_resources()
        finally:
            self._close_sync_resources()
            self._telemetry_bridge.flush_and_close()
            if self._background_loop is not None:
                self._background_loop.close()
                self._background_loop = None

    async def _close_async_resources(self) -> None:
        if self._mcp_client is not None:
            try:
                await self._mcp_client.close_async()
            except Exception:
                pass
            finally:
                self._mcp_tools_synced = False

        provider_close_async = getattr(self._provider, "close_async", None)
        if callable(provider_close_async):
            try:
                await provider_close_async()
            except Exception:
                pass

    def _close_sync_resources(self) -> None:
        provider_close = getattr(self._provider, "close", None)
        if callable(provider_close):
            try:
                provider_close()
            except Exception:
                pass

    def __del__(self) -> None:
        if getattr(self, "_background_loop", None) is not None:
            try:
                self._background_loop.close()
            except Exception:
                pass

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()
