"""Retry configuration and exponential backoff calculations for LLM requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import errno
import random
import time


def is_network_or_provider_error(err: BaseException) -> bool:
    """Determine whether an exception represents a transient network or provider error."""
    # Never retry program exit or task cancellation
    if isinstance(err, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False

    # 1. Standard library network and timeout exceptions
    if isinstance(err, (ConnectionError, TimeoutError)):
        return True

    # 2. Check HTTP status code if present (e.g. 408, 429, 500, 502, 503, 504)
    status_code = (
        getattr(err, "status_code", None)
        or getattr(err, "code", None)
        or getattr(err, "http_status", None)
    )
    if isinstance(status_code, int) and status_code in {408, 429, 500, 502, 503, 504}:
        return True

    # 3. Known network socket errors
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

    # 4. Class hierarchy name keywords for network and provider SDK errors
    keywords = (
        "Timeout",
        "Connection",
        "Connect",
        "RateLimit",
        "Overload",
        "Unavailable",
        "RemoteDisconnected",
        "APIConnection",
        "APITimeout",
        "InternalServer",
    )
    class_names = {c.__name__ for c in type(err).__mro__}
    if any(any(kw in name for kw in keywords) for name in class_names):
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

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {self.max_delay}")
        if self.backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {self.backoff_factor}")

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


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate the exponential backoff delay in seconds for a zero-indexed retry attempt.

    Attempt 0 corresponds to the first retry after initial failure.
    """
    delay = config.initial_delay * (config.backoff_factor ** attempt)
    capped_delay = min(delay, config.max_delay)
    if config.jitter and capped_delay > 0:
        return random.uniform(0, capped_delay)
    return capped_delay
