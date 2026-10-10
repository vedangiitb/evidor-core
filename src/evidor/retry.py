"""Retry configuration and exponential backoff calculations for LLM requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import email.utils
import errno
import math
import random
import time
from typing import Any


def extract_retry_after(err: BaseException) -> float | None:
    """Extract Retry-After delay in seconds from an exception or its HTTP response if present."""
    # 1. Direct attribute on exception (e.g. err.retry_after)
    direct_val = getattr(err, "retry_after", None)
    if isinstance(direct_val, (int, float)):
        return max(0.0, float(direct_val))
    if isinstance(direct_val, str):
        try:
            return max(0.0, float(direct_val.strip()))
        except ValueError:
            pass

    # 2. Check headers on exception or response
    headers: Any = getattr(err, "headers", None) or getattr(getattr(err, "response", None), "headers", None)
    if isinstance(headers, Mapping):
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val is not None:
            if isinstance(val, (int, float)):
                return max(0.0, float(val))
            if isinstance(val, str):
                s = val.strip()
                try:
                    return max(0.0, float(s))
                except ValueError:
                    # Try parsing as HTTP-date (RFC 7231)
                    try:
                        date_dt = email.utils.parsedate_to_datetime(s)
                        now_dt = datetime.now(timezone.utc)
                        delta = (date_dt - now_dt).total_seconds()
                        return max(0.0, delta)
                    except Exception:
                        pass

    return None


def is_network_or_provider_error(err: BaseException) -> bool:
    """Determine whether an exception represents a transient network or provider error.

    Explicitly excludes permanent programming errors (e.g. ValueError, TypeError)
    and process termination signals. Avoids loose substring name matching to prevent
    misclassifying configuration or permanent exceptions.
    """
    # Never retry program exit or task cancellation
    if isinstance(err, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False

    # Never retry standard programming errors
    if isinstance(err, (TypeError, ValueError, KeyError, AttributeError, IndexError, SyntaxError, ZeroDivisionError, AssertionError, NotImplementedError)):
        return False

    # 1. Standard library network and timeout exceptions
    if isinstance(err, (ConnectionError, TimeoutError)):
        return True

    # 2. Standard library socket / network OSError with transient errnos
    if isinstance(err, OSError) and getattr(err, "errno", None) in {
        errno.ECONNRESET,
        errno.ECONNREFUSED,
        errno.ECONNABORTED,
        errno.ETIMEDOUT,
        errno.ENETUNREACH,
        errno.ENETDOWN,
        errno.EPIPE,
    }:
        return True

    # 3. Check HTTP status code if present (e.g. 408, 429, 500, 502, 503, 504)
    status_code = (
        getattr(err, "status_code", None)
        or getattr(err, "code", None)
        or getattr(err, "http_status", None)
        or getattr(getattr(err, "response", None), "status_code", None)
    )
    if isinstance(status_code, int) and status_code in {408, 429, 500, 502, 503, 504}:
        return True

    # 4. Known provider SDK and HTTP library transient exception types
    mod_name = getattr(type(err), "__module__", "") or ""
    cls_name = type(err).__name__

    # OpenAI & Anthropic SDK transient errors
    if mod_name.startswith(("openai", "anthropic")):
        if cls_name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "OverloadedError",
        }:
            return True

    # HTTPX & Requests & urllib3 transient errors
    if mod_name.startswith(("httpx", "requests", "urllib3", "http.client", "urllib.error")):
        if cls_name in {
            "TransportError",
            "TimeoutException",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "NetworkError",
            "RemoteDisconnected",
            "ConnectionError",
            "Timeout",
        }:
            return True

    # Google GenAI transient errors
    if mod_name.startswith("google.genai"):
        if cls_name == "APIError" and isinstance(status_code, int) and status_code in {408, 429, 500, 502, 503, 504}:
            return True

    return False


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryConfig:
    """Configuration for retrying failed LLM requests with exponential backoff.

    By default, only transient network and provider errors are retried.
    Users can override retryable exceptions by specifying ``retryable_exceptions``.
    """

    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] | None = None
    sleep_fn: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)
    sleep_async_fn: Callable[[float], Any] = field(default=asyncio.sleep, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate max_retries: must be int and not bool
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError(
                f"max_retries must be an integer, got {type(self.max_retries).__name__}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

        # Validate numeric bounds and reject non-finite numbers
        for name, val, min_val in [
            ("initial_delay", self.initial_delay, 0.0),
            ("max_delay", self.max_delay, 0.0),
            ("backoff_factor", self.backoff_factor, 1.0),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(val).__name__}")
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")
            if val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")

        if self.max_delay < self.initial_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= initial_delay ({self.initial_delay})"
            )

        if not isinstance(self.jitter, bool):
            raise TypeError(f"jitter must be a boolean, got {type(self.jitter).__name__}")

        if self.retryable_exceptions is not None:
            if not isinstance(self.retryable_exceptions, tuple):
                raise TypeError(
                    f"retryable_exceptions must be a tuple of exception classes or None, got {type(self.retryable_exceptions).__name__}"
                )
            for exc in self.retryable_exceptions:
                if not (isinstance(exc, type) and issubclass(exc, BaseException)):
                    raise TypeError(
                        f"Each element in retryable_exceptions must be a subclass of BaseException, got {exc}"
                    )

        if not callable(self.sleep_fn):
            raise TypeError(f"sleep_fn must be callable, got {type(self.sleep_fn).__name__}")
        if not callable(self.sleep_async_fn):
            raise TypeError(f"sleep_async_fn must be callable, got {type(self.sleep_async_fn).__name__}")

    @property
    def is_enabled(self) -> bool:
        """Return True if retries are enabled."""
        return self.max_retries > 0

    def is_retryable(self, err: BaseException) -> bool:
        """Return True if the exception should be retried under this configuration."""
        if not self.is_enabled:
            return False
        if self.retryable_exceptions is not None:
            return isinstance(err, self.retryable_exceptions)
        return is_network_or_provider_error(err)


DEFAULT_RETRY_CONFIG = RetryConfig(max_retries=3)


def calculate_backoff_delay(
    attempt: int, config: RetryConfig, err: BaseException | None = None
) -> float:
    """Calculate backoff delay in seconds for a zero-indexed retry attempt.

    If the error provides a Retry-After hint, that delay is respected up to config.max_delay.
    Otherwise, calculates exponential backoff with optional jitter.
    """
    if err is not None:
        retry_after = extract_retry_after(err)
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, config.max_delay)

    delay = config.initial_delay * (config.backoff_factor ** attempt)
    capped_delay = min(delay, config.max_delay)
    if config.jitter and capped_delay > 0:
        return random.uniform(0, capped_delay)
    return capped_delay
