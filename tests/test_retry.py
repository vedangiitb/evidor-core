"""Tests for LLM retries, exponential backoff, configuration, and telemetry integration."""

import asyncio
from typing import Any
import pytest

from evidor import (
    Agent,
    GenerationRequest,
    GenerationResponse,
    InMemorySink,
    RetryConfig,
    calculate_backoff_delay,
    is_network_or_provider_error,
)
from evidor.retry import DEFAULT_RETRY_CONFIG
from evidor.telemetry import (
    AgentRunEndEvent,
    AgentRunStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    PrometheusSink,
    TraceContext,
    semconv,
)


class FlakyProvider:
    """Mock provider that fails a specified number of times before succeeding."""

    def __init__(self, fail_count: int, error_type: type[BaseException] = ConnectionError) -> None:
        self.model = "flaky-model"
        self.fail_count = fail_count
        self.error_type = error_type
        self.attempts = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise self.error_type(f"Failure attempt {self.attempts}")
        return GenerationResponse(text=f"Success on attempt {self.attempts}", model=self.model)

    def with_model(self, model: str) -> "FlakyProvider":
        return self


# ==============================================================================
# RetryConfig & Backoff Calculation Tests
# ==============================================================================


def test_is_network_or_provider_error() -> None:
    class ConnectionConfigurationError(Exception):
        """Permanent configuration error whose name contains 'Connection'."""
        pass

    class MockHTTPStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class MockOpenAIRateLimit(Exception):
        pass
    MockOpenAIRateLimit.__module__ = "openai"
    MockOpenAIRateLimit.__name__ = "RateLimitError"

    class MockAnthropicInternalServer(Exception):
        pass
    MockAnthropicInternalServer.__module__ = "anthropic"
    MockAnthropicInternalServer.__name__ = "InternalServerError"

    # Standard network & timeout errors
    assert is_network_or_provider_error(ConnectionError("lost")) is True
    assert is_network_or_provider_error(TimeoutError("timed out")) is True

    # Transient HTTP status codes
    assert is_network_or_provider_error(MockHTTPStatusError(429)) is True
    assert is_network_or_provider_error(MockHTTPStatusError(503)) is True
    assert is_network_or_provider_error(MockHTTPStatusError(500)) is True
    assert is_network_or_provider_error(MockHTTPStatusError(502)) is True
    assert is_network_or_provider_error(MockHTTPStatusError(504)) is True
    assert is_network_or_provider_error(MockHTTPStatusError(408)) is True

    # Provider SDK transient types
    assert is_network_or_provider_error(MockOpenAIRateLimit("rate limited")) is True
    assert is_network_or_provider_error(MockAnthropicInternalServer("server error")) is True

    # Similarly named permanent errors must NOT be retried (Issue 2)
    assert is_network_or_provider_error(ConnectionConfigurationError("bad config")) is False

    # Non-network/provider errors are NOT retryable
    assert is_network_or_provider_error(ValueError("bad value")) is False
    assert is_network_or_provider_error(TypeError("wrong type")) is False
    assert is_network_or_provider_error(KeyError("missing key")) is False
    assert is_network_or_provider_error(MockHTTPStatusError(400)) is False
    assert is_network_or_provider_error(MockHTTPStatusError(401)) is False
    assert is_network_or_provider_error(MockHTTPStatusError(403)) is False
    assert is_network_or_provider_error(MockHTTPStatusError(404)) is False
    assert is_network_or_provider_error(KeyboardInterrupt()) is False



def test_retry_config_defaults() -> None:
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.initial_delay == 0.5
    assert config.max_delay == 60.0
    assert config.backoff_factor == 2.0
    assert config.jitter is True
    assert config.is_enabled is True
    assert config.retryable_exceptions is None
    assert config.is_retryable(ConnectionError("lost")) is True
    assert config.is_retryable(TimeoutError("timed out")) is True
    assert config.is_retryable(ValueError("bad value")) is False



def test_retry_config_validation() -> None:
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        RetryConfig(max_retries=-1)

    with pytest.raises(ValueError, match="initial_delay must be >= 0"):
        RetryConfig(initial_delay=-0.5)

    with pytest.raises(ValueError, match="max_delay must be >= 0"):
        RetryConfig(max_delay=-1.0)

    with pytest.raises(ValueError, match="backoff_factor must be >= 1.0"):
        RetryConfig(backoff_factor=0.9)


def test_calculate_backoff_delay_without_jitter() -> None:
    config = RetryConfig(
        initial_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        jitter=False,
    )
    # Attempt 0: initial_delay * (2^0) = 1.0
    assert calculate_backoff_delay(0, config) == 1.0
    # Attempt 1: initial_delay * (2^1) = 2.0
    assert calculate_backoff_delay(1, config) == 2.0
    # Attempt 2: initial_delay * (2^2) = 4.0
    assert calculate_backoff_delay(2, config) == 4.0
    # Attempt 3: initial_delay * (2^3) = 8.0
    assert calculate_backoff_delay(3, config) == 8.0
    # Attempt 4: capped at max_delay = 10.0
    assert calculate_backoff_delay(4, config) == 10.0


def test_calculate_backoff_delay_with_jitter() -> None:
    config = RetryConfig(
        initial_delay=2.0,
        backoff_factor=2.0,
        max_delay=10.0,
        jitter=True,
    )
    for attempt in range(5):
        delay = calculate_backoff_delay(attempt, config)
        max_possible = min(config.initial_delay * (config.backoff_factor ** attempt), config.max_delay)
        assert 0.0 <= delay <= max_possible


# ==============================================================================
# Agent Retry Configuration & Disabling Tests
# ==============================================================================


def test_agent_retry_config_defaults() -> None:
    provider = FlakyProvider(fail_count=0)
    agent = Agent(provider)
    assert agent.retry_config.max_retries == 3
    assert agent.retry_config.is_enabled is True


def test_agent_retry_config_override_max_retries() -> None:
    provider = FlakyProvider(fail_count=0)
    agent = Agent(provider, max_retries=5)
    assert agent.retry_config.max_retries == 5
    assert agent.retry_config.is_enabled is True


def test_agent_retry_config_custom_instance() -> None:
    provider = FlakyProvider(fail_count=0)
    custom = RetryConfig(max_retries=2, initial_delay=0.1, backoff_factor=1.5, jitter=False)
    agent = Agent(provider, retry_config=custom)
    assert agent.retry_config == custom


def test_agent_retry_config_custom_instance_with_max_retries_override() -> None:
    provider = FlakyProvider(fail_count=0)
    custom = RetryConfig(max_retries=2, initial_delay=0.1)
    agent = Agent(provider, retry_config=custom, max_retries=4)
    assert agent.retry_config.max_retries == 4
    assert agent.retry_config.initial_delay == 0.1


def test_agent_disable_retries_via_max_retries_zero() -> None:
    provider = FlakyProvider(fail_count=0)
    agent = Agent(provider, max_retries=0)
    assert agent.retry_config.max_retries == 0
    assert agent.retry_config.is_enabled is False


def test_agent_disable_retries_via_none() -> None:
    provider = FlakyProvider(fail_count=0)
    agent = Agent(provider, retry_config=None)
    assert agent.retry_config.max_retries == 0
    assert agent.retry_config.is_enabled is False


def test_agent_disable_retries_via_false() -> None:
    provider = FlakyProvider(fail_count=0)
    agent = Agent(provider, retry_config=False)
    assert agent.retry_config.max_retries == 0
    assert agent.retry_config.is_enabled is False


def test_agent_enable_retries_via_true() -> None:
    provider = FlakyProvider(fail_count=0)
    agent = Agent(provider, retry_config=True)
    assert agent.retry_config.max_retries == 3


def test_agent_invalid_retry_config_type() -> None:
    provider = FlakyProvider(fail_count=0)
    with pytest.raises(TypeError, match="Invalid retry_config"):
        Agent(provider, retry_config="invalid")  # type: ignore[arg-type]


# ==============================================================================
# Retry Execution & Telemetry Flow Tests
# ==============================================================================


def test_retry_successful_recovery() -> None:
    captured_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=3,
        initial_delay=0.01,
        backoff_factor=2.0,
        jitter=False,
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    provider = FlakyProvider(fail_count=2)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    try:
        response = agent.send("Hello agent")
        assert response.text == "Success on attempt 3"
        assert provider.attempts == 3
    finally:
        agent.close()

    # Backoff sleep was called twice
    assert len(captured_sleeps) == 2
    assert captured_sleeps[0] == 0.01
    assert captured_sleeps[1] == 0.02

    # Check telemetry events
    llm_starts = sink.filter(LLMCallStartEvent)
    assert len(llm_starts) == 3
    assert llm_starts[0].retry_attempt == 0
    assert llm_starts[1].retry_attempt == 1
    assert llm_starts[2].retry_attempt == 2

    llm_ends = sink.filter(LLMCallEndEvent)
    assert len(llm_ends) == 3
    assert llm_ends[0].status == "error"
    assert llm_ends[0].retry_attempt == 0
    assert llm_ends[1].status == "error"
    assert llm_ends[1].retry_attempt == 1
    assert llm_ends[2].status == "ok"
    assert llm_ends[2].retry_attempt == 2
    assert llm_ends[2].to_attributes()[semconv.EVIDOR_RETRY_COUNT] == 2

    retry_errors = sink.filter(ErrorEvent)
    # 2 intermediate retry errors
    assert len(retry_errors) == 2
    assert retry_errors[0].outcome == "retrying"
    assert retry_errors[0].retry_count == 1
    assert retry_errors[0].details["retry_delay_seconds"] == 0.01
    assert retry_errors[0].details["provider"] == "FlakyProvider"

    assert retry_errors[1].outcome == "retrying"
    assert retry_errors[1].retry_count == 2
    assert retry_errors[1].details["retry_delay_seconds"] == 0.02

    # Overall agent run succeeded
    agent_ends = sink.filter(AgentRunEndEvent)
    assert len(agent_ends) == 1
    assert agent_ends[0].status == "ok"


def test_retry_exhausted_raises_exception() -> None:
    captured_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=2,
        initial_delay=0.01,
        backoff_factor=2.0,
        jitter=False,
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    provider = FlakyProvider(fail_count=10)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    try:
        with pytest.raises(ConnectionError, match="Failure attempt 3"):
            agent.send("Hello fail")
    finally:
        agent.close()

    # 1 initial + 2 retries = 3 total attempts
    assert provider.attempts == 3
    assert len(captured_sleeps) == 2

    llm_starts = sink.filter(LLMCallStartEvent)
    assert len(llm_starts) == 3

    llm_ends = sink.filter(LLMCallEndEvent)
    assert len(llm_ends) == 3
    assert all(e.status == "error" for e in llm_ends)

    errors = sink.filter(ErrorEvent)
    # Intermediate retry errors (2) + exhausted error (1) + agent run error (1)
    retrying_errors = [e for e in errors if e.outcome == "retrying"]
    assert len(retrying_errors) == 2
    assert retrying_errors[0].retry_count == 1
    assert retrying_errors[1].retry_count == 2

    exhausted_errors = [e for e in errors if e.outcome == "exhausted"]
    assert len(exhausted_errors) == 1
    assert exhausted_errors[0].retry_count == 2
    assert exhausted_errors[0].details["max_retries"] == 2

    agent_ends = sink.filter(AgentRunEndEvent)
    assert len(agent_ends) == 1
    assert agent_ends[0].status == "error"


def test_disabled_retries_fails_immediately() -> None:
    captured_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=0,
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    provider = FlakyProvider(fail_count=5)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    try:
        with pytest.raises(ConnectionError, match="Failure attempt 1"):
            agent.send("Run once")
    finally:
        agent.close()

    assert provider.attempts == 1
    assert len(captured_sleeps) == 0

    llm_starts = sink.filter(LLMCallStartEvent)
    assert len(llm_starts) == 1
    assert llm_starts[0].retry_attempt == 0

    llm_ends = sink.filter(LLMCallEndEvent)
    assert len(llm_ends) == 1
    assert llm_ends[0].status == "error"

    errors = sink.filter(ErrorEvent)
    # Initial failure outcome is "failed"
    llm_error = next(e for e in errors if e.details.get("provider") == "FlakyProvider")
    assert llm_error.outcome == "failed"
    assert llm_error.retry_count == 0


def test_default_does_not_retry_non_network_errors() -> None:
    captured_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=3,
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    # Provider raises ValueError (a programming error, not a network/provider error)
    provider = FlakyProvider(fail_count=5, error_type=ValueError)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    try:
        with pytest.raises(ValueError, match="Failure attempt 1"):
            agent.send("Don't retry ValueError")
    finally:
        agent.close()

    # Must fail on first attempt without retrying
    assert provider.attempts == 1
    assert len(captured_sleeps) == 0


def test_custom_retryable_exceptions_overrides_default() -> None:
    captured_sleeps: list[float] = []
    # Explicitly configure ValueError to be retryable
    config = RetryConfig(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        retryable_exceptions=(ValueError,),
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    provider = FlakyProvider(fail_count=1, error_type=ValueError)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    try:
        response = agent.send("Retry custom error")
        assert response.text == "Success on attempt 2"
    finally:
        agent.close()

    assert provider.attempts == 2
    assert len(captured_sleeps) == 1



def test_non_retryable_exception_fails_immediately() -> None:
    captured_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=3,
        retryable_exceptions=(TimeoutError,),
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    provider = FlakyProvider(fail_count=5, error_type=ValueError)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    try:
        with pytest.raises(ValueError, match="Failure attempt 1"):
            agent.send("Don't retry ValueError")
    finally:
        agent.close()

    assert provider.attempts == 1
    assert len(captured_sleeps) == 0


def test_async_send_with_retries() -> None:
    captured_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        sleep_fn=captured_sleeps.append,
    )

    sink = InMemorySink()
    provider = FlakyProvider(fail_count=1)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)

    async def _run() -> None:
        try:
            response = await agent.send_async("Async test")
            assert response.text == "Success on attempt 2"
        finally:
            await agent.close_async()

    asyncio.run(_run())
    assert provider.attempts == 2
    assert len(captured_sleeps) == 1

    llm_ends = sink.filter(LLMCallEndEvent)
    assert len(llm_ends) == 2
    assert llm_ends[0].status == "error"
    assert llm_ends[1].status == "ok"


# ==============================================================================
# Adapter Telemetry Verification Tests
# ==============================================================================


def test_prometheus_retry_metrics() -> None:
    from prometheus_client import CollectorRegistry

    registry = CollectorRegistry()
    sink = PrometheusSink(registry=registry)
    config = RetryConfig(
        max_retries=2,
        initial_delay=0.001,
        jitter=False,
        sleep_fn=lambda _: None,
    )

    provider = FlakyProvider(fail_count=1)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)
    try:
        response = agent.send("Prometheus retry test")
        assert response.text == "Success on attempt 2"
    finally:
        agent.close()

    # Verify evidor_llm_retries_total incremented
    retries_metric = registry.get_sample_value(
        "evidor_llm_retries_total",
        {"provider": "FlakyProvider", "model": "flaky-model"},
    )
    assert retries_metric == 1.0

    # Verify exposition text contains metric
    text = sink.export_text()
    assert "evidor_llm_retries_total" in text


def test_opentelemetry_retry_span_events() -> None:
    from tests.test_opentelemetry_adapter import create_test_tracer
    from evidor.telemetry.adapters.opentelemetry import OpenTelemetrySink

    tracer_provider, exporter = create_test_tracer()
    sink = OpenTelemetrySink(tracer_provider=tracer_provider)
    config = RetryConfig(
        max_retries=2,
        initial_delay=0.001,
        jitter=False,
        sleep_fn=lambda _: None,
    )

    provider = FlakyProvider(fail_count=1)
    agent = Agent(provider=provider, retry_config=config, telemetry=sink)
    try:
        response = agent.send("Otel retry test")
        assert response.text == "Success on attempt 2"
    finally:
        agent.close()

    spans = exporter.get_finished_spans()
    # 2 LLM call spans + 1 agent.run span
    llm_spans = [s for s in spans if s.name.startswith("llm.")]
    assert len(llm_spans) == 2

    # First LLM span was an error and has a retry event
    failed_span = llm_spans[0]
    event_names = [e.name for e in failed_span.events]
    assert "retry" in event_names
    retry_ev = next(e for e in failed_span.events if e.name == "retry")
    assert retry_ev.attributes[semconv.EVIDOR_RETRY_COUNT] == 1
    assert retry_ev.attributes[semconv.EVIDOR_RETRY_OUTCOME] == "retrying"


# ==============================================================================
# Dedicated Tests for Issues 1-7
# ==============================================================================


def test_retry_config_strict_validation() -> None:
    # Issue 7: Booleans rejected for max_retries
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        RetryConfig(max_retries=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        RetryConfig(max_retries=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        RetryConfig(max_retries=3.5)  # type: ignore[arg-type]

    # Non-finite numbers rejected
    with pytest.raises(ValueError, match="initial_delay must be a finite number"):
        RetryConfig(initial_delay=float("inf"))
    with pytest.raises(ValueError, match="initial_delay must be a finite number"):
        RetryConfig(initial_delay=float("nan"))
    with pytest.raises(ValueError, match="max_delay must be a finite number"):
        RetryConfig(max_delay=float("inf"))
    with pytest.raises(ValueError, match="backoff_factor must be a finite number"):
        RetryConfig(backoff_factor=float("nan"))

    # max_delay must be >= initial_delay
    with pytest.raises(ValueError, match="must be >= initial_delay"):
        RetryConfig(initial_delay=10.0, max_delay=5.0)

    # retryable_exceptions validation
    with pytest.raises(TypeError, match="retryable_exceptions must be a tuple"):
        RetryConfig(retryable_exceptions=[ValueError])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Each element in retryable_exceptions must be a subclass"):
        RetryConfig(retryable_exceptions=("not_an_exc",))  # type: ignore[arg-type]


def test_extract_retry_after_and_backoff_delay_honors_hint() -> None:
    # Issue 3: Rate-limit and Retry-After delay extraction and honoring
    from datetime import datetime, timezone, timedelta
    import email.utils
    from evidor.retry import extract_retry_after

    class CustomRateLimit(Exception):
        def __init__(self, retry_after: Any = None, headers: Any = None) -> None:
            if retry_after is not None:
                self.retry_after = retry_after
            if headers is not None:
                self.headers = headers

    # 1. Direct attribute numeric and str
    err1 = CustomRateLimit(retry_after=5.0)
    assert extract_retry_after(err1) == 5.0

    err2 = CustomRateLimit(retry_after="12.5")
    assert extract_retry_after(err2) == 12.5

    # 2. Header dictionary
    err3 = CustomRateLimit(headers={"Retry-After": "3.5"})
    assert extract_retry_after(err3) == 3.5

    err4 = CustomRateLimit(headers={"retry-after": 7})
    assert extract_retry_after(err4) == 7.0

    # 3. RFC 7231 HTTP-date
    future_time = datetime.now(timezone.utc) + timedelta(seconds=20)
    date_str = email.utils.format_datetime(future_time)
    err5 = CustomRateLimit(headers={"Retry-After": date_str})
    val = extract_retry_after(err5)
    assert val is not None
    assert 17.0 <= val <= 23.0

    # 4. calculate_backoff_delay honors retry_after hint
    config = RetryConfig(initial_delay=0.1, max_delay=10.0, jitter=False)
    # Without err hint: 0.1 * 2^0 = 0.1
    assert calculate_backoff_delay(0, config) == 0.1
    # With err hint of 3.5s:
    assert calculate_backoff_delay(0, config, err=err3) == 3.5
    # Capped at max_delay:
    err_large = CustomRateLimit(retry_after=999.0)
    assert calculate_backoff_delay(0, config, err=err_large) == 10.0


@pytest.mark.asyncio
async def test_async_sleep_cancellation_during_backoff() -> None:
    # Issue 1: Non-blocking async sleep cancels cleanly without tying up thread pool
    provider = FlakyProvider(fail_count=5)
    config = RetryConfig(max_retries=3, initial_delay=5.0, jitter=False)
    agent = Agent(provider=provider, retry_config=config)

    task = asyncio.create_task(agent.send_async("hello"))
    # Allow task to fail attempt 0 and enter non-blocking asyncio.sleep
    await asyncio.sleep(0.05)
    assert not task.done()
    # Cancel the task while in backoff sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_summarization_retries_sync() -> None:
    # Issue 4: Summarization retries with unified retry policy and telemetry (sync)
    main_provider = FlakyProvider(fail_count=0)
    flaky_summary = FlakyProvider(fail_count=1)

    sink = InMemorySink()
    config = RetryConfig(max_retries=2, initial_delay=0.001, jitter=False, sleep_fn=lambda _: None)
    agent = Agent(
        provider=main_provider,
        summarization_model=flaky_summary,
        context_window=200,
        max_messages=3,
        retry_config=config,
        telemetry=sink,
    )

    try:
        agent.send("Message 1: " + "a" * 100)
        agent.send("Message 2: " + "b" * 100)
        resp = agent.send("Message 3: " + "c" * 100)
        assert resp.text
    finally:
        agent.close()

    assert flaky_summary.attempts == 2
    error_events = sink.filter(ErrorEvent)
    assert any(e.outcome == "retrying" and e.details.get("provider") == "FlakyProvider" for e in error_events)


@pytest.mark.asyncio
async def test_summarization_retries_async() -> None:
    # Issue 4: Summarization retries with unified retry policy and telemetry (async)
    main_provider = FlakyProvider(fail_count=0)
    flaky_summary = FlakyProvider(fail_count=1)

    sink = InMemorySink()
    captured_async_sleeps: list[float] = []
    config = RetryConfig(
        max_retries=2,
        initial_delay=0.001,
        jitter=False,
        sleep_async_fn=captured_async_sleeps.append,
    )
    agent = Agent(
        provider=main_provider,
        summarization_model=flaky_summary,
        context_window=200,
        max_messages=3,
        retry_config=config,
        telemetry=sink,
    )

    try:
        await agent.send_async("Message 1: " + "a" * 100)
        await agent.send_async("Message 2: " + "b" * 100)
        resp = await agent.send_async("Message 3: " + "c" * 100)
        assert resp.text
    finally:
        await agent.close_async()

    assert flaky_summary.attempts == 2
    assert len(captured_async_sleeps) == 1
    error_events = sink.filter(ErrorEvent)
    assert any(e.outcome == "retrying" and e.details.get("provider") == "FlakyProvider" for e in error_events)


def test_retry_boundary_narrowed_to_provider_call() -> None:
    # Issue 5: Retry boundary narrowed strictly to provider call;
    # response processing / telemetry errors do NOT re-trigger provider.generate!
    class CorruptedResponse:
        text = "Generated text"
        model = "corrupted-model"

        @property
        def tool_calls(self) -> Any:
            # Simulate transient network/connection error raised during response parsing
            raise ConnectionError("Connection dropped during response parsing")

    class CorruptedProvider:
        def __init__(self) -> None:
            self.attempts = 0
            self.model = "corrupted-model"

        def generate(self, request: GenerationRequest) -> Any:
            self.attempts += 1
            return CorruptedResponse()

        def with_model(self, model: str) -> Any:
            return self

    provider = CorruptedProvider()
    config = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False, sleep_fn=lambda _: None)
    agent = Agent(provider=provider, retry_config=config)  # type: ignore[arg-type]

    # When response.tool_calls raises ConnectionError, it is OUTSIDE provider.generate try-block.
    # It must raise immediately without retrying the provider.
    with pytest.raises(ConnectionError, match="Connection dropped during response parsing"):
        agent.send("Hello")

    # Provider was called only ONCE, proving retry boundary is narrowed strictly to provider.generate!
    assert provider.attempts == 1

